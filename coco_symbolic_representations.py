import json
import pandas as pd


def read_jsons(paths):
    jsons = []
    for path in paths:
        jsons.append(json.load(open(path, "r")))

    return jsons


def jsons_to_df(jsons, json_type):
    assert json_type in ["captions", "instances", "categories"], "json_type must be in [captions, instances, categories]"

    df_list = []
    if json_type == "captions":
        for json in jsons:
            df_list.append(pd.json_normalize(json, "annotations")[["image_id", "caption"]])
    elif json_type == "instances":
        for json in jsons:
            df_list.append(pd.json_normalize(json, "annotations")[["image_id", "bbox", "category_id"]])
    else:
        for json in jsons:
            df_list.append(pd.json_normalize(json, "categories")[["id", "name"]])

    return pd.concat(df_list)


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


    captions_df["image"] = captions_df["image_id"].apply(
        lambda x: format(x, "012d") + ".jpg")

    instances_df["image"] = instances_df["image_id"].apply(
        lambda x: format(x, "012d") + ".jpg")


    captions_df = captions_df.groupby("image").agg({"caption": lambda x: "\n".join(x)}).reset_index()

    instances_df = pd.merge(instances_df, instance_categories_df, left_on="category_id", right_on="id")

    instances_df["bbox"] = instances_df.apply(
        lambda row: row["name"] + ": " + str(row["bbox"]), axis=1,
        )

    instances_df = instances_df.groupby("image").agg({"bbox": lambda x: "\n".join(x)}).reset_index()


    # (123287, 122218)
    symbolic_rep_df = pd.merge(captions_df, instances_df, on="image", how="left")
    # 123287
    symbolic_rep_df.to_pickle('symbolic_representation_coco_trainval_2017.pkl')

