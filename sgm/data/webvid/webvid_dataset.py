import os

import pandas as pd

from .base_video_dataset import TextVideoDataset


class WebVid(TextVideoDataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """

    def _load_metadata(self):
        assert self.metadata_folder_name is not None
        assert self.cut is not None
        metadata_dir = os.path.join(self.metadata_dir, self.metadata_folder_name)
        if self.key is None:
            metadata_fp = os.path.join(
                metadata_dir, f"results_{self.cut}_{self.split}.csv"
            )
        else:
            metadata_fp = os.path.join(
                metadata_dir, f"results_{self.cut}_{self.split}_{self.key}.csv"
            )
        print(metadata_fp)
        metadata = pd.read_csv(
            metadata_fp,
            on_bad_lines="skip",
            encoding="ISO-8859-1",
            engine="python",
            sep=",",
        )

        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        elif self.split == "val":
            try:
                metadata = metadata.sample(1000, random_state=0)
            except:
                print(
                    "there are less than 1000 samples in the val set, thus no downsampling is done"
                )
                pass

        metadata["caption"] = metadata["name"]
        del metadata["name"]
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = str(sample["videoid"]) + ".mp4"
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        if not os.path.exists(full_video_fp):
            full_video_fp = os.path.join(self.data_dir, "videos", rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample["caption"]


if __name__ == "__main__":
    from tqdm import tqdm
    import imageio
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--motion_scale", type=int, default=4)
    opt = parser.parse_known_args()[0]

    def write_text_to_file(text, file_path):
        with open(file_path, "w") as file:
            file.write(text)

    config = {
        "dataset_name": "WebVid",
        "data_dir": "/msra_data/videos_rmwm",
        "metadata_dir": "/msra_data",
        "split": "val",
        "cut": "2M",
        "key": "wmrm_all",
        "subsample": 1,
        "text_params": {"input": "text"},
        "video_params": {
            "input_res_h": 320,  # todo: check the input_res_h
            "input_res_w": 320,  # todo: check the input_res_w
            "tsfm_params": {
                "norm_mean": [0.5, 0.5, 0.5],
                "norm_std": [0.5, 0.5, 0.5],
            },
            "num_frames": opt.num_frames,
            "prop_factor": 30,
            "loading": "lax",
        },
        "metadata_folder_name": "webvid10m_meta",
        "first_stage_key": "jpg",
        "cond_stage_key": "txt",
        "skip_missing_files": False,
    }

    dataset = WebVid(**config)
    length = dataset.__len__()

    txt_out_path = os.path.join(
        opt.out_path, f"num{opt.num_frames}_ms{opt.motion_scale}", "txt"
    )
    video_out_high_path = os.path.join(
        opt.out_path, f"num{opt.num_frames}_ms{opt.motion_scale}", "videoHigh"
    )
    video_out_low_path = os.path.join(
        opt.out_path, f"num{opt.num_frames}_ms{opt.motion_scale}", "videoLow"
    )
    os.makedirs(txt_out_path, exist_ok=True)
    os.makedirs(video_out_high_path, exist_ok=True)
    os.makedirs(video_out_low_path, exist_ok=True)

    for idx in tqdm(range(length)):
        print(idx)
        item = dataset.__getitem__(idx)
        video = item["jpg"]
        txt = item["txt"]

        video_new = (
            ((video.transpose(3, 1) * 0.5 + 0.5).clamp(0, 1) * 255.0)
            .numpy()
            .astype(np.uint8)
        )
        video_list = [img for img in video_new]
        imageio.mimsave(
            os.path.join(video_out_high_path, f"{idx:09d}.gif"),
            video_list,
            duration=1,
            loops=1,
        )
        imageio.mimsave(
            os.path.join(video_out_low_path, f"{idx:09d}.gif"),
            video_list[:: opt.motion_scale],
            duration=1 * opt.motion_scale,
            loops=1,
        )

        write_text_to_file(txt, os.path.join(txt_out_path, f"{idx:09d}.txt"))
