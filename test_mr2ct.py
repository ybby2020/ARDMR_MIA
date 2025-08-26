import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from evaluation import evaluation as mt


from model.ARDMR import DMR
def main():
    test_dir = '' #dataset path
    # model_idx = -1
    weights = [10,2]
    model_folder = 'multi_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    img_size = (64, 160, 160)
    model = DMR(inchannel=1, statechannel=8, vol_shape=img_size)
    model.cuda()

    test_epoch = 'latest'
    model_name = 'reg-Epoch_{}.pth'.format(test_epoch)
    # typ = ''
    # savepth = 'results/' + model_folder + model_name + f'{typ}'
    # if not os.path.exists(savepth):
    #     os.makedirs(savepth)

    best_model = torch.load(os.path.join(model_dir,model_name))
    model.load_state_dict(best_model)
    model.cuda()

    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    test_composed = transforms.Compose([
        trans.NumpyType((np.float32, np.float32)),
                                        ])
    test_set = datasets.SJTUTestliverDataset(test_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            x = data[0].cuda()
            y = data[1].cuda()
            x_seg = data[2].cuda()
            y_seg = data[3].cuda()
            output = model(x,y)

            #eva
            def_out = reg_model([x_seg.cuda().float(), output[3].cuda()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(output[3].detach().cpu().numpy()[0, :, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            dsc = mt.compute_dice(def_out.long(),y_seg.long())[0]
            dsc_raw = mt.compute_dice(x_seg.long(),y_seg.long())[0]

            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc,dsc_raw))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            eval_dsc_def.update(dsc.item() if isinstance(dsc, torch.Tensor) else dsc , x.size(0))
            eval_dsc_raw.update(dsc_raw.item() if isinstance(dsc_raw, torch.Tensor) else dsc_raw , x.size(0))

        print('Deformed DSC: {:.3f} +- {:.3f}, RAW DSC: {:.3f} +- {:.3f}'.
              format(eval_dsc_def.avg,
                            eval_dsc_def.std,
                            eval_dsc_raw.avg,
                            eval_dsc_raw.std,
                     ))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()