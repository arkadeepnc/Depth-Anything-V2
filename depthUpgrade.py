import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib.pyplot as plt
from transformers import pipeline
import open3d as o3d
from kornia.geometry import depth_to_3d, depth_to_normals
from skimage.segmentation import felzenszwalb, quickshift, slic
from skimage.segmentation import mark_boundaries
from PIL import Image
from scipy.stats import mode
from tqdm import tqdm
from kornia.filters import MedianBlur, Laplacian
from kornia.feature import LoFTR
from registerPoints import _align_camera_centers, _apply_similarity_transform
import pdb
# generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)

def getMonoDepthModel():
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cuda:0'))
    model.cuda()
    model.eval()
    return model


def writePCD(depth, rgb_image, fname = None):

    # point grey camera
    cam_mtx = np.array([[       1553.9663302771853,      0.0,       339.47357153102087    ],
                        [      0.0,      1556.191376668082,      288.44960314558057    ],
                        [      0.0,       0.0,       1.0    ]  ]) 

    # realsense
    # cam_mtx = np.array([[       605.84136963,      0.0,       314.27282715    ],
    #                     [      0.0,      605.50720215,      254.17677307   ],
    #                     [      0.0,       0.0,       1.0    ]  ]) 

    points = depth_to_3d (torch.from_numpy(depth[None, None, ...]),camera_matrix= torch.from_numpy(cam_mtx[ None, ...]),normalize_points= False).squeeze()
    x, y, z = points[0,...].flatten(), points[1,...].flatten(), points[2,...].flatten()
    normals = depth_to_normals( depth = torch.from_numpy(depth[None, None, ...]),camera_matrix= torch.from_numpy(cam_mtx[ None, ...])).squeeze()
    nx, ny, nz = normals[0,...].flatten(), normals[1,...].flatten(), normals[2,...].flatten()
    # plt.imshow((normals.transpose(2,0).cpu().numpy()+1)/2)
    # plt.imsave('./normals.png', (normals.permute(1,2,0).cpu().numpy()+1)/2)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)/255.0
    r,g,b = rgb_image[...,0].flatten(), rgb_image[...,1].flatten(), rgb_image[...,2].flatten()
    if fname:
        pcd = o3d.geometry.PointCloud()
        pts = torch.stack([x,y,z], dim=0).detach().cpu().numpy()
        normals = torch.stack([nx,ny,nz], dim=0).detach().cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(pts.T)
        pcd.normals = o3d.utility.Vector3dVector(normals.T)
        pcd.colors = o3d.utility.Vector3dVector(np.vstack([r,g,b]).T)
        o3d.io.write_point_cloud('./'+fname+'.ply', pcd)

    return {'points':points, 'normals': normals, 'rgb' : rgb_image}


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    # print(mask.shape, target.shape, prediction.shape,'<---??')
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

# def upgradeDepth():



def conventionalSegmentation(predicted_depth, metric_depth, segments):

    scaled_depth = np.zeros_like(metric_depth)
    scale_all= []
    shift_all = []
    weights_all = []
    num_segments = np.max(segments)
    true_depth_t = torch.tensor(metric_depth[None,...], dtype=torch.float32, device='cuda:0', requires_grad=False)
    pred_depth_t = torch.tensor(predicted_depth[None,...], dtype=torch.float32, device='cuda:0', requires_grad=False)
    out_depth_t = torch.zeros_like(true_depth_t) 
    # out_depth = np.zeros_like(true_depth) 
    for id in tqdm(range(num_segments)):
        mask = (segments == id)
        mask_t = torch.tensor(mask[None,...], dtype=torch.float32, device='cuda:0', requires_grad=False)
        scale, shift = compute_scale_and_shift(pred_depth_t, true_depth_t, mask_t)
        # out_depth_t = scale.view(-1, 1, 1) * pred_depth_t + shift.view(-1, 1, 1)
        # out_depth_t += scale.view(-1, 1, 1) * pred_depth_t * mask_t + shift.view(-1, 1, 1)
        scale_all.append(scale.cpu().numpy())
        shift_all.append(shift.cpu().numpy())
        weights_all.append(mask.sum())

    mode_scale = np.average(np.array(scale_all).squeeze(), weights = np.array(weights_all))
    mode_shift = np.average(np.array(shift_all).squeeze(), weights = np.array(weights_all))
    print(f"mode scele {mode_scale} mode shift {mode_shift}")
    print(scale_all[-1], shift_all[-1])
    out_depth = predicted_depth * mode_scale + mode_shift

    plt.imshow(out_depth)
    plt.show()
    return out_depth

def SAMSegmentation(predicted_depth, metric_depth, masks):

    scaled_depth = np.zeros_like(metric_depth)
    scale_all= []
    shift_all = []
    weights_all = []
    # num_segments = np.max(segments)
    true_depth_t = torch.tensor(metric_depth[None,...], dtype=torch.float32, device='cuda:0', requires_grad=False)
    pred_depth_t = torch.tensor(predicted_depth[None,...], dtype=torch.float32, device='cuda:0', requires_grad=False)
    out_depth_t = torch.zeros_like(true_depth_t) 
    # out_depth = np.zeros_like(true_depth) 
    # for id in tqdm(range(num_segments)):
    for mask in tqdm(masks):
        # mask = (segments == id)
        mask_t = torch.tensor(mask[None,...], dtype=torch.float32, device='cuda:0', requires_grad=False)
        scale, shift = compute_scale_and_shift(pred_depth_t, true_depth_t, mask_t)
        # out_depth_t = scale.view(-1, 1, 1) * pred_depth_t + shift.view(-1, 1, 1)
        # out_depth_t += scale.view(-1, 1, 1) * pred_depth_t * mask_t + shift.view(-1, 1, 1)
        scale_all.append(scale.cpu().numpy())
        shift_all.append(shift.cpu().numpy())
        weights_all.append(mask.sum())

    mode_scale = np.average(np.array(scale_all).squeeze(), weights = np.array(weights_all))
    mode_shift = np.average(np.array(shift_all).squeeze(), weights = np.array(weights_all))
    # mode_scale = mode(np.array(scale_all).squeeze())[0] 
    # mode_shift = mode(np.array(shift_all).squeeze())[0] 
    print(f"mode scele {mode_scale} mode shift {mode_shift}")
    print(scale_all[-1], shift_all[-1])
    out_depth = predicted_depth * mode_scale + mode_shift

    plt.imshow(out_depth)
    plt.show()

    plt.plot(np.array(scale_all), label = 'scales')
    # plt.plot(np.array(shift_all), label = 'shifts')
    plt.legend()
    plt.show()
    
    return out_depth

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    # gc.collect()

def show_masks_on_image(raw_image, masks):
  plt.imshow(np.array(raw_image))
  ax = plt.gca()
  ax.set_autoscale_on(False)
  for mask in masks:
      show_mask(mask, ax=ax, random_color=True)
  plt.axis("off")
  plt.show()

# def registerPCD(dir):
#     import glob
#     fnames = glob.glob(dir)

def RANSAC_align_camera_centers(colmap_camera_centers, robot_cam_centers, max_iter = 2500, min_point_err = 0.5):
    num_images, dim = colmap_camera_centers.shape[0], colmap_camera_centers.shape[1]
    best_align_t_R = None
    best_align_t_T = None
    best_align_t_s = None
    best_err = 1e9
    max_inliers = 4
    # pdb.set_trace()
    # for it_ in tqdm(range(max_iter), desc=f'Curr error = {best_err}'):
    for it_ in  tqdm(range(max_iter), desc=f'error {best_err}'):
        # confirmed_inliers = []
        maybe_inliers = torch.randperm(num_images)[:4]
        maybe_inliers_colmap = colmap_camera_centers[maybe_inliers,:]
        maybe_inliers_robot = robot_cam_centers[maybe_inliers,:]
        maybe_model = _align_camera_centers(maybe_inliers_colmap, maybe_inliers_robot)
        maybe_inliers_colmap_aligned = _apply_similarity_transform(colmap_camera_centers[None,...],maybe_model[0], maybe_model[1], maybe_model[2][None])
        # pdb.set_trace()
        # test
        per_point_error = torch.linalg.norm(maybe_inliers_colmap_aligned - robot_cam_centers, dim = -1).flatten()
        # print(per_point_error,'<---')
        confirmed_inliers = torch.where(per_point_error<min_point_err)[0]
        
        if confirmed_inliers.shape[0] >=max_inliers:
            # pdb.set_trace()
            confirmed_inliers_colmap = colmap_camera_centers[confirmed_inliers,:]
            confirmed_inliers_robot =      robot_cam_centers[confirmed_inliers,:]
            confirmed_model = _align_camera_centers(confirmed_inliers_colmap,confirmed_inliers_robot)
            confirmed_inliers_colmap_aligned = _apply_similarity_transform(confirmed_inliers_colmap[None,...],\
                                                                           confirmed_model[0], 
                                                                           confirmed_model[1],
                                                                           confirmed_model[2][None])
            per_point_error_confirmed = torch.linalg.norm(confirmed_inliers_colmap_aligned - confirmed_inliers_robot, dim = -1).flatten().sum()

            if (per_point_error_confirmed < best_err) or (len(confirmed_inliers) > max_inliers):
                max_inliers = confirmed_inliers.shape[0]
                print(f'per_point_error {per_point_error_confirmed} << best error {best_err}, # inliers: {len(confirmed_inliers)} ')
                best_err = per_point_error_confirmed
                best_align_t_R = confirmed_model[0]
                best_align_t_T = confirmed_model[1]
                best_align_t_s = confirmed_model[2]
                
    
    return best_align_t_R, best_align_t_T, best_align_t_s                




if __name__ == "__main__":

    
    model = getMonoDepthModel()
    blur = MedianBlur((9,9))

    # image_fnames = ['/home/atkesonlab/multLightWorkspace/data/reflective_1/images/left_0.png',\
    #                 '/home/atkesonlab/multLightWorkspace/data/reflective_1/images/left_1.png',\
    #                 '/home/atkesonlab/multLightWorkspace/data/reflective_1/images/left_2.png',\
    #                 '/home/atkesonlab/multLightWorkspace/data/reflective_1/images/right_0.png',\
    #                 '/home/atkesonlab/multLightWorkspace/data/reflective_1/images/right_1.png',\
    #                 '/home/atkesonlab/multLightWorkspace/data/reflective_1/images/right_2.png']

    image_fnames = ['/home/atkesonlab/multLightWorkspace/data/trash/images/left_0.png',\
                     '/home/atkesonlab/multLightWorkspace/data/trash/images/right_0.png']

    # image_fnames = ['./rs_1.png',\
    #                  './rs_2.png']
    
    # depth_fnames = ['/home/atkesonlab/multLightWorkspace/data/reflective_1/depths/left_0_depth.npy',\
    #                 '/home/atkesonlab/multLightWorkspace/data/reflective_1/depths/left_1_depth.npy',\
    #                 '/home/atkesonlab/multLightWorkspace/data/reflective_1/depths/left_2_depth.npy',\
    #                 '/home/atkesonlab/multLightWorkspace/data/reflective_1/depths/right_0_depth.npy',\
    #                 '/home/atkesonlab/multLightWorkspace/data/reflective_1/depths/right_1_depth.npy',\
    #                 '/home/atkesonlab/multLightWorkspace/data/reflective_1/depths/right_2_depth.npy']

    depth_fnames = ['/home/atkesonlab/multLightWorkspace/data/trash/depths/left_0_depth.npy',\
                    '/home/atkesonlab/multLightWorkspace/data/trash/depths/right_0_depth.npy']

    scaled_PCDs = []
    for i, (image_fname, depth_fname) in enumerate(zip(image_fnames, depth_fnames)):
        raw_img = cv2.imread(image_fname)
        # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        pred_depth = model.infer_image(raw_img) # HxW raw depth map in numpy

 

        ## conventional segmentation
        # segments = quickshift(raw_img, kernel_size=21, max_dist=6, ratio=0.5)
        segments = slic(raw_img, n_segments = 100, compactness = 20, sigma = 0)
        plt.imshow(mark_boundaries(raw_img, segments))
        plt.show()
        metric_depth = np.load(depth_fname) #* 0.001
        input = torch.tensor(metric_depth[None,None,...])
        output = blur(blur(blur(blur(blur(input)))))
        metric_depth = input.cpu().numpy().squeeze()
        pred_depth_upgraded = conventionalSegmentation(pred_depth, metric_depth, segments)
        out = writePCD(pred_depth_upgraded, raw_img, str(i))
        scaled_PCDs.append(out)
        _ = writePCD(metric_depth, raw_img, 'metric_depth')
        writePCD(pred_depth, raw_img, 'raw_preds')
        # exit()

        ## SAM segmentation
        # generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)
        # raw_img_PIL = Image.open(image_fname).convert("RGB")
        # outputs = generator(raw_img_PIL, points_per_batch=1)
        # masks = outputs["masks"]
        # scores = outputs["scores"]
        # show_masks_on_image(raw_img, masks)
        # metric_depth = np.load(depth_fname) 
        # pred_depth_upgraded_sam = SAMSegmentation(pred_depth, metric_depth, masks)
        # # writePCD(pred_depth_upgraded_sam, raw_img, 'pred_depth_upgraded_SAM')
        # out = writePCD(pred_depth_upgraded_sam, raw_img, str(i))
        # scaled_PCDs.append(out)
        # exit()

    img_tmp = cv2.imread(image_fnames[0],0)/255.0
    img_1 = cv2.imread(image_fnames[1],0)/255.0

    input_ = {'image0': torch.from_numpy(img_tmp[None, None, ...]).float(), \
              'image1': torch.from_numpy(img_1[None, None, ...]).float()}
    loftr = LoFTR('indoor')
    out = loftr(input_)
    # pdb.set_trace()
    kp1 = out['keypoints0']
    kp2 = out['keypoints1']
    img = np.hstack([img_tmp, img_1])
    plt.imshow(img, cmap = 'gray')
    for i in range(kp1.shape[0]):
        tmp_x = kp1[i,1].numpy()
        tmp_y = kp1[i,0].numpy()

        img_x = kp2[i,1] + 768
        img_y = kp2[i,0]
        plt.plot([tmp_x, img_x], [tmp_y, img_y])
    
    plt.show()

    pcd_src = scaled_PCDs[0]['points']
    pcd_dst = scaled_PCDs[1]['points']
    confidence = out['confidence']>0.5
    kp_tmp = (kp1[confidence,:]).int()
    kp_dst = (kp2[confidence,:]).int()
    # pdb.set_trace()
    pcd_src_corresp = pcd_src[:,kp_tmp[:,1], kp_tmp[:,0]]
    pcd_dst_corresp = pcd_dst[:,kp_dst[:,1], kp_dst[:,0]]

    align_t_R, align_t_T, align_t_s = RANSAC_align_camera_centers(pcd_dst_corresp.T, pcd_src_corresp.T)
    dst_points_reshaped = pcd_dst.permute(1,2,0).reshape(-1,3)
    aligned_dst_points = _apply_similarity_transform(dst_points_reshaped[None,...], align_t_R, align_t_T, align_t_s[None]).squeeze()
    aligned_src_points = pcd_src.permute(1,2,0).reshape(-1,3)
    src_color = scaled_PCDs[0]['rgb'].reshape(-1,3)
    src_normals = scaled_PCDs[0]['normals'].T
    dst_color = scaled_PCDs[1]['rgb'].reshape(-1,3)
    dst_normals = scaled_PCDs[1]['normals'].T
    # make pcd
    def make_pcd(points, colors, normals, name): 
        # pdb.set_trace()
        pcd = o3d.geometry.PointCloud()
        pts = points.detach().cpu().numpy()
        normals = normals
        normals = normals
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud('./'+name+'.ply', pcd)
    make_pcd(aligned_src_points, src_color, src_normals,'src')
    make_pcd(aligned_dst_points, dst_color, dst_normals,'dst')





    













