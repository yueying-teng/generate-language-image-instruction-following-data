import json
import pandas as pd


def read_jsons(paths):
    jsons = []
    for path in paths:
        jsons.append(json.load(open(path, "r")))

    return jsons


def jsons_to_df(jsons, json_type):
    assert json_type in [
        "captions", "instances", "categories", "images_info"
        ], \
            "json_type must be in [captions, instances, categories, images_info]"

    df_list = []
    if json_type == "captions":
        for json in jsons:
            df_list.append(pd.json_normalize(json, "annotations")[["image_id", "caption"]])
    elif json_type == "instances":
        for json in jsons:
            df_list.append(pd.json_normalize(json, "annotations")[["image_id", "bbox", "category_id"]])
    elif json_type == "images_info":
        for json in jsons:
            df_list.append(pd.json_normalize(json, "images")[["file_name", "height", "width"]])
    else:
        # categories in val and train are the same
        df_list.append(pd.json_normalize(jsons[0], "categories")[["id", "name"]])

    return pd.concat(df_list)


def process_captions_df(captions_df):

    captions_df["image"] = captions_df["image_id"].apply(
        lambda x: format(x, "012d") + ".jpg")

    captions_df = captions_df.groupby("image").agg({"caption": lambda x: "\n".join(x)}).reset_index()

    return captions_df


def process_instances_df(instances_df, instance_categories_df, images_info_df):

    instances_df["image"] = instances_df["image_id"].apply(
        lambda x: format(x, "012d") + ".jpg")

    instances_df = pd.merge(instances_df, instance_categories_df, left_on="category_id", right_on="id")

    instances_df = pd.merge(instances_df, images_info_df, left_on="image", right_on="file_name", how="left")

    # change bbox format from [x, y, width, height] to [xmin, xmax, ymin, ymax]
    # where (x, y) is the top-left corner of the bounding box.
    # width and height are the width and height of the bounding box
    # e.g.
    # xmin, ymin, bboxw, bboxh = [0.0, 264.94, 16.49, 43.16]
    # xmin, xmax, ymin, ymax = [xmin/imgw, (xmin + bboxw)/imgw, ymin/imgh, (ymin + bboxh)/imgh]
    instances_df["bbox"] = instances_df.apply(
        lambda row: row["name"] + ": " + str([
            round(row["bbox"][0]/ row["width"], 3),
            round((row["bbox"][0] + row["bbox"][2]) / row["width"], 3),
            round(row["bbox"][1]/ row["height"], 3),
            round((row["bbox"][1] + row["bbox"][3]) / row["height"], 3),
            ]),
        axis=1,
    )

    instances_df = instances_df.groupby("image").agg({"bbox": lambda x: "\n".join(x)}).reset_index()

    return instances_df


if __name__ == "__main__":
    captions_paths = [
        "COCO2017/annotations/captions_train2017.json",
        "COCO2017/annotations/captions_val2017.json",
    ]

    instances_paths = [
        "COCO2017/annotations/instances_train2017.json",
        "COCO2017/annotations/instances_val2017.json",
    ]

    captions_jsons = read_jsons(captions_paths)
    instances_jsons = read_jsons(instances_paths)

    captions_df = jsons_to_df(captions_jsons, "captions")
    instances_df = jsons_to_df(instances_jsons, "instances")
    instance_categories_df = jsons_to_df(instances_jsons, "categories")
    images_info_df = jsons_to_df(instances_jsons, "images_info")

    captions_df = process_captions_df(captions_df)
    instances_df = process_instances_df(instances_df, instance_categories_df, images_info_df)

    # (123287, 122218)
    symbolic_rep_df = pd.merge(captions_df, instances_df, on="image", how="left")
    # 123287
    symbolic_rep_df.to_pickle('symbolic_representation_coco_trainval_2017.pkl')

