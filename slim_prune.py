from models import *
from utils.utils import *
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.prune_utils import *
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='sparse model weights')
    parser.add_argument('--global_percent', type=float, default=0.8, help='global channel prune percent')
    parser.add_argument('--layer_keep', type=float, default=0.01, help='channel keep percent per layer')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)

    if opt.weights.endswith(".pt"):
        model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
    else:
        _ = load_darknet_weights(model, opt.weights)
    print('\nloaded weights from ',opt.weights)

    #eval_model测试模型，调用test函数
    eval_model = lambda model:test(model=model,cfg=opt.cfg, data=opt.data)
    #obtain_num_parameters 获得模型的参数数量
    obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

    print("\nlet's test the original model first:")
    # no_grad 不跟踪计算梯度
    with torch.no_grad():
        origin_model_metric = eval_model(model)
    origin_nparameters = obtain_num_parameters(model)

    CBL_idx, Conv_idx, prune_idx, _, _= parse_module_defs2(model.module_defs)


    # 将BN层的权重系数得到，如某一层有32个卷积层，则有32个BN系数，待分析？？
    bn_weights = gather_bn_weights(model.module_list, prune_idx)
    # 对bn_weights进行从小到大排序，根据global_percent计算bn阈值thresh
    # sorted_index为bn的权重在未排序列表中的位置
    sorted_bn = torch.sort(bn_weights)[0]
    sorted_bn, sorted_index = torch.sort(bn_weights)
    thresh_index = int(len(bn_weights) * opt.global_percent)
    thresh = sorted_bn[thresh_index].cuda()

    print(f'Global Threshold should be less than {thresh:.4f}.')




    #%%
    def obtain_filters_mask(model, thresh, CBL_idx, prune_idx):
        '''
        功能：根据thre值确定含有BN层的卷积中那些层需要剪去

        '''

        pruned = 0
        total = 0
        num_filters = []
        filters_mask = []
        for idx in CBL_idx:
            bn_module = model.module_list[idx][1]
            if idx in prune_idx:
                #获得bn权重的绝对值存在weight_copy中
                weight_copy = bn_module.weight.data.abs().clone()
                #channels为BN的权重个数，即卷积层的特征层数
                channels = weight_copy.shape[0] #
                #min_channel_num 表示某卷积层需要保留的最小feature map 层数，eg channels=32 则int(channels * opt.layer_keep)=0
                #此时min_channel_num=1，即卷积中最少有一层的feature map输出
                min_channel_num = int(channels * opt.layer_keep) if int(channels * opt.layer_keep) > 0 else 1
                # mask BN权值大于thresh时 mask对应的值设为1，否则设为0
                mask = weight_copy.gt(thresh).float()
                #如果根据BN的权值得到保留层数少于需要保留的最小层数，则将BN weight 降序排列，取前面的min_channel_num个
                if int(torch.sum(mask)) < min_channel_num:
                    _, sorted_index_weights = torch.sort(weight_copy,descending=True)
                    mask[sorted_index_weights[:min_channel_num]]=1.
                remain = int(mask.sum())
                #pruned 为剪枝掉的层个数
                pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                        f'remaining channel: {remain:>4d}')
            else:
                mask = torch.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]

            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.clone())

        prune_ratio = pruned / total
        print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

        return num_filters, filters_mask

    num_filters, filters_mask = obtain_filters_mask(model, thresh, CBL_idx, prune_idx)
    # CBLidx2mask 为字典类型，层名：每层的mask，mask为0表示减掉，为1表示保留
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
    # CBLidx2filters 为字典类型，层名：每层保留的filter个数
    CBLidx2filters = {idx: filters for idx, filters in zip(CBL_idx, num_filters)}

    for i in model.module_defs:
        if i['type'] == 'shortcut':
            i['is_access'] = False

    print('merge the mask of layers connected to shortcut!')
    merge_mask(model, CBLidx2mask, CBLidx2filters)



    def prune_and_eval(model, CBL_idx, CBLidx2mask):
        # 根据mask将对应的BN层gamma设置为0或者原值
        model_copy = deepcopy(model)

        for idx in CBL_idx:
            bn_module = model_copy.module_list[idx][1]
            mask = CBLidx2mask[idx].cuda()
            bn_module.weight.data.mul_(mask)

        with torch.no_grad():
            mAP = eval_model(model_copy)[0][2]

        print(f'mask the gamma as zero, mAP of the model is {mAP:.4f}')


    prune_and_eval(model, CBL_idx, CBLidx2mask)

    ##将mask值转换成numpy格式
    for i in CBLidx2mask:
        CBLidx2mask[i] = CBLidx2mask[i].clone().cpu().numpy()



    pruned_model = prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask)
    print("\nnow prune the model but keep size,(actually add offset of BN beta to following layers), let's see how the mAP goes")

    with torch.no_grad():
        eval_model(pruned_model)

    for i in model.module_defs:
        if i['type'] == 'shortcut':
            i.pop('is_access')

    compact_module_defs = deepcopy(model.module_defs)
    # 将compact_module_defs模型的卷积参数量设为剪枝后的数量
    for idx in CBL_idx:
        assert compact_module_defs[idx]['type'] == 'convolutional'
        compact_module_defs[idx]['filters'] = str(CBLidx2filters[idx])


    compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs, (img_size, img_size)).to(device)
    compact_nparameters = obtain_num_parameters(compact_model)

    init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)


    random_input = torch.rand((1, 3, img_size, img_size)).to(device)

    def obtain_avg_forward_time(input, model, repeat=200):

        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time, output

    print('testing inference time...')
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)


    print('testing the final model...')
    with torch.no_grad():
        compact_model_metric = eval_model(compact_model)


    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", f'{origin_model_metric[0][2]:.6f}', f'{compact_model_metric[0][2]:.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)



    pruned_cfg_name = opt.cfg.replace('/', f'/prune_{opt.global_percent}_keep_{opt.layer_keep}_')
    pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
    print(f'Config file has been saved: {pruned_cfg_file}')

    compact_model_name = opt.weights.replace('/', f'/prune_{opt.global_percent}_keep_{opt.layer_keep}_')
    if compact_model_name.endswith('.pt'):
        compact_model_name = compact_model_name.replace('.pt', '.weights')
    save_weights(compact_model, path=compact_model_name)
    print(f'Compact model has been saved: {compact_model_name}')
