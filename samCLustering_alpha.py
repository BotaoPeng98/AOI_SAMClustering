import glob
import sys, os
import time
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import random, tqdm

import operator
from functools import reduce

sys.path.append(".")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from torchvision.transforms import ToPILImage
from sklearn.cluster import AgglomerativeClustering
from models import CLIPModel


def find_bounding_contours(mask):
    """
    Find bounding contours of mask
    """
    mask = mask.astype(np.uint8)
    bounding_contours = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_contours.append([x, y, w, h])
    return bounding_contours


def build_point_grid(n_per_side: int, image) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    image_points = image.copy()
    crop_blur = cv2.GaussianBlur(image_points, (5, 5), 0)
    image_points2 = image.copy()
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points_candidates = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    points = []
    for point in points_candidates:
        x, y = point
        a, b = int(x * image.shape[1]), int(y * image.shape[0])
        cv2.circle(image_points2, (a, b), 2, (0, 0, 255), 1)
        if crop_blur[int(x * crop_blur.shape[0]), int(y * crop_blur.shape[1])] < 255:
            points.append((x, y))
            cv2.circle(image_points, (a, b), 2, (0, 0, 255), 1)
        else:
            points.append((x + 5, y + 5))
            cv2.circle(image_points, (a + 5, b + 5), 2, (0, 0, 255), 1)
    return points_candidates


def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int, image
):
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points, image))
    return points_by_layer


def build_adaptive_points_grid(
    image
):
    image_c = image.copy()
    # get lines from image
    blurred = cv2.GaussianBlur(image_c, (5, 5), 0)
    candidate_points = []
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # w_scale = random.uniform(0, 1)
        # h_scale = random.uniform(0, 1)
        w_scale = 0.5
        h_scale = 0.5
        w1_scale = random.uniform(0, 0.2)
        w2_scale = random.uniform(0.8, 0.9)
        h1_scale = random.uniform(0, 0.2)
        h2_scale = random.uniform(0.8, 0.9)
        cv2.circle(
            image_c, (int(x + w * w1_scale), int(y + h * h1_scale)), 2, (0, 0, 255), 1
        )
        cv2.circle(
            image_c, (int(x + w * w1_scale), int(y + h * h2_scale)), 2, (0, 0, 255), 1
        )
        cv2.circle(
            image_c, (int(x + w * w2_scale), int(y + h * h1_scale)), 2, (0, 0, 255), 1
        )
        cv2.circle(
            image_c, (int(x + w * w2_scale), int(y + h * h2_scale)), 2, (0, 0, 255), 1
        )
        cv2.circle(
            image_c, (int(x + w * w_scale), int(y + h * h_scale)), 2, (0, 0, 255), 1
        )
        candidate_points.append(
            [
                int(x + w * w1_scale) / image.shape[1],
                int(y + h * h1_scale) / image.shape[0],
            ]
        )
        candidate_points.append(
            [
                int(x + w * w1_scale) / image.shape[1],
                int(y + h * h2_scale) / image.shape[0],
            ]
        )
        candidate_points.append(
            [
                int(x + w * w2_scale) / image.shape[1],
                int(y + h * h1_scale) / image.shape[0],
            ]
        )
        candidate_points.append(
            [
                int(x + w * w2_scale) / image.shape[1],
                int(y + h * h2_scale) / image.shape[0],
            ]
        )
        candidate_points.append(
            [
                int(x + w * w_scale) / image.shape[1],
                int(y + h * h_scale) / image.shape[0],
            ]
        )
    return [candidate_points]


def save_crop_(
    downsampled_image,
    anns,
    index=0,
    start_coord=(0, 0),
    full_mask=None,
    save_ok=False,
    save_pth=None,
):

    patch_weighted = downsampled_image.copy()
    if len(anns) == 0 or len(anns) == 1:
        return full_mask
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    # anns_no_1st = [[i["bbox"][0]+i["bbox"][2]//2, i["bbox"][1]+i["bbox"][3]//2] for i in sorted_anns[1:]]

    anns_no_1st = [i["segmentation"] for i in sorted_anns[1:]]
    mask = reduce(operator.or_, anns_no_1st).astype(np.uint8)
    if len(mask.shape) == 2:
        mask = mask[:, :, np.newaxis]
    bgr_mask = np.concatenate([np.zeros_like(mask), np.zeros_like(mask), mask], axis=2)

    patch_weighted = cv2.addWeighted(patch_weighted, 1, bgr_mask * 255, 0.5, 0)

    full_mask[
        start_coord[0] : start_coord[0] + 1024,
        start_coord[1] : start_coord[1] + 1024,
    ] += bgr_mask

    if save_ok and save_pth:
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)
        cv2.imwrite(f"{save_pth}/{index}_0_all_.jpg", patch_weighted)
    return full_mask


def crop_via_mask(image, mask, save_ok, save_pth):
    if save_ok and save_pth:
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)
    mask = mask.astype(np.uint8)[:, :, -1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in tqdm.tqdm(enumerate(contours), desc="Saving crops"):
        x, y, w, h = cv2.boundingRect(contour)
        crop_o = image[y : y + h, x : x + w]
        if save_ok and save_pth:
            crop = crop_o.copy()
            cv2.imwrite(f"{save_pth}/{i}.jpg", crop)


def export_mask_to_json(
    img_pth,
    mask,
    anno_width,
    anno_height,
    mask_json_pth,
):
    cur_mask = []
    mask = mask.astype(np.uint8)[:, :, -1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(
            contour
        )  # (x,y) is the left-top corner of the rectangle
        # 对检测出的边缘芯粒做一下过滤（因为是准备训练数据，下一步数据增广的时候，会生成边缘数据）
        x_center = x + w // 2
        y_center = y + h // 2
        if (
            x_center < anno_width // 2
            or x_center > mask.shape[1] - anno_width // 2
            or y_center < anno_height // 2
            or y_center > mask.shape[0] - anno_height // 2
        ):
            continue
        cur_mask.append(
            {
                "id": -1,
                "meta": {
                    "coor": [x_center, y_center, anno_width, anno_height],
                    "type": 1,
                },
            }
        )

    with open(mask_json_pth, "r") as f:
        mediate_json = json.load(f)
    mediate_json["annotations"].extend(
        [
            {
                "filename": img_pth,
                "labels": cur_mask,
            }
        ]
    )
    with open(mask_json_pth, "w") as f:
        json.dump(mediate_json, f, ensure_ascii=False, indent=4)

def hierarchyClustering(X, cluster_num):
    print("Perform Hierarchical clustering...")
    X = X.astype(np.float32)
    hierarchy = AgglomerativeClustering(n_clusters=cluster_num, metric='cosine', linkage='complete')
    hierarchy.fit(X)
    labels = hierarchy.labels_
    return labels


if __name__ == "__main__":

    ## TODO: load a input of width and height of a chiplet
    input_width = 78
    input_height = 78

    # img_dir info
    wafer_type = "05EXL"
    img_dirs_pth = f"./testset"
    
    img_dirs_lst = os.listdir(img_dirs_pth)
    detect_light = "1"  # assign which light to detect
    cluster_nums = 30
    assert detect_light in [
        "1",
        "2",
        "3",
        "4",
    ], "detect light should be assigned as 1, 2, 3, 4"

    
    # load sam model
    sam_checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # initialize json file
    samClusterJSON = f"./output/sam_mask_cluster.json"

    if not os.path.exists(samClusterJSON.replace("sam_mask_cluster.json", "")):
        os.makedirs(samClusterJSON.replace("sam_mask_cluster.json", ""))
    categories = [{"name": str(i), "id": i} for i in range(cluster_nums)]
    with open(samClusterJSON, "w") as f:
        init_json = {
            "categories": [categories],
            "annotations": [],
        }
        json.dump(init_json, f, ensure_ascii=False, indent=4)

    t_start = time.time()
    img_pth_lst = glob.glob(img_dirs_pth + f"/IMAGE{detect_light}*.jpg")
    for img_pth in tqdm.tqdm(img_pth_lst, desc=f"Segment chiplets in {img_dirs_pth}"):
        img_name = img_pth.split("/")[-1].split(".")[0]
        image_l = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)
        image_rgb = cv2.cvtColor(image_l, cv2.COLOR_GRAY2RGB)
        blurred_image_rgb = cv2.GaussianBlur(image_rgb, (5, 5), 0)

        patch_x = [i for i in range(0, image_l.shape[1], 1024)]
        patch_y = [i for i in range(0, image_l.shape[0], 1024)]
        image_patchs = {}

        img_id = 0

        # patchfy an image
        for x in patch_x:
            for y in patch_y:
                image_patch = image_l[y : y + 1024, x : x + 1024]
                image_patch_rgb = cv2.cvtColor(image_patch, cv2.COLOR_GRAY2RGB)
                blurred_patch = cv2.GaussianBlur(image_patch_rgb, (5, 5), 0)
                image_patchs[img_id] = {"img": blurred_patch, "start_coord": (y, x)}
                img_id += 1

        mask_full = np.zeros(image_rgb.shape, np.uint8)

        for idx, img_p in image_patchs.items():
            points_ = build_adaptive_points_grid(img_p["img"])
            if len(points_[0]) == 0:
                continue
            mask_generator_2 = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=None,
                pred_iou_thresh=0.2,
                stability_score_thresh=0.9,
                crop_n_layers=0,
                crop_n_points_downscale_factor=0,
                point_grids=points_,
                min_mask_region_area=1000000,
            )
            patch_tensor = (
                torch.from_numpy(img_p["img"])
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(device)
            )
            masks = mask_generator_2.generate(img_p["img"])
            if len(masks) == 0:
                continue

            mask_full = save_crop_(
                img_p["img"],
                masks,
                index=idx,
                start_coord=img_p["start_coord"],
                full_mask=mask_full,
                save_ok=False,
                save_pth=f"results/{model_type}/{img_name}",
            )

        ## mask vis
        one_mat = np.zeros_like(mask_full)
        one_mat[:, :, 2] = 1
        bgr_mask = np.where(mask_full > 0, one_mat, 0)
        image_mask_vis = cv2.addWeighted(image_rgb, 1, bgr_mask * 255, 0.5, 0)
        
        ## Modify save_crop_ to output a json file which contains the mask center and corresponding image id
        export_mask_to_json(
            img_pth=img_pth,
            mask=mask_full,
            anno_width=input_width,
            anno_height=input_height,
            mask_json_pth=samClusterJSON,
        )
    time_cost_det = (time.time() - t_start) / 60
    print(f"Chiplet detection time cost: {time_cost_det:.2f} min")

    
    t_start = time.time()
    # load clip pretrained model
    clip_pth = "./checkpoints/ViT-L-14-336px.pt"
    model = CLIPModel(model_name=clip_pth).cuda()
    # model.print_params()
    model.eval()
    
    # read annotations
    trainset = []
    with open(samClusterJSON, "r") as f:
        sam_json = json.load(f)
    for annotation in sam_json["annotations"]:
        img_pth = annotation["filename"]
        img = cv2.imread(img_pth)
        for label in annotation["labels"]:
            x, y, w, h = label["meta"]["coor"]
            crop = img[y - h // 2 : y + h // 2, x - w // 2 : x + w // 2]
            trainset.append(crop)
    
    # load dataset and extract features
    dataloader_train = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
    features = []
    to_pil = ToPILImage()
    for iteration, x in tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Inferring image features and labels"):
        x = x.permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            x_ = model.preprocess(to_pil(x[0])).unsqueeze(0).cuda()
            feature = model.encode_image(x_)
        features.append(feature.cpu().numpy())
    features = np.concatenate(features, axis=0)
    # print("Feature shape:", features.shape)
    
    images_embedding = features / np.linalg.norm(features, axis=1, keepdims=True)
    preds = hierarchyClustering(images_embedding, cluster_nums)
    
    # write clustering results back to json
    img_id = 0
    with open(samClusterJSON, "r") as f:
        sam_json = json.load(f)
    for annotation in sam_json["annotations"]:
        img_pth = annotation["filename"]
        img = cv2.imread(img_pth)
        for label in annotation["labels"]:
            label["id"] = int(preds[img_id])
            img_id += 1
    with open(samClusterJSON, "w") as f:
        json.dump(sam_json, f, ensure_ascii=False, indent=4)
    
    time_cost_clustering = (time.time() - t_start) / 60
    print(f"Clustering time cost: {time_cost_clustering:.2f} min")
    total_time_cost = time_cost_det + time_cost_clustering
    print(f"Whole time cost: {total_time_cost:.2f} min")
    
    
    
    
    
