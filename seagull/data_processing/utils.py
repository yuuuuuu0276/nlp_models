from functools import partial
from typing import List, Dict, Any, Union, Optional, Tuple

import datasets

from datasets import disable_caching
disable_caching()


def get_torch_dataset(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset.set_format(type="torch")
    return dataset


def _merge_scene_uncanny_caption(
    data_instances: Dict[str, List[Any]],
    scene_colname_and_special_token: Tuple[str, str],
    uncanny_colname_and_special_token: Tuple[str, str],
    caption_colname_and_special_token: Tuple[str, str],
    end_of_caption_special_token: str,
    merge_colname: str,
) -> Dict[str, List[Any]]:
    """See: https://pages.github.coecis.cornell.edu/cs4740/hw4-fa23/seagull.data_processing.utils.html."""
    new_lst = []

    l = len(data_instances[scene_colname_and_special_token[0]])
    for i in range(l):
      cur = []
      cur.append(scene_colname_and_special_token[1])
      cur.append(data_instances[scene_colname_and_special_token[0]][i])
      cur.append(uncanny_colname_and_special_token[1])
      cur.append(data_instances[uncanny_colname_and_special_token[0]][i])

      if caption_colname_and_special_token[0] in data_instances.keys() :
        cur.append(caption_colname_and_special_token[1])
        cur.append(data_instances[caption_colname_and_special_token[0]][i])
        cur.append(end_of_caption_special_token)
      else:
        cur.append(caption_colname_and_special_token[1])

      new_lst.append(" ".join(cur))

    data_instances[merge_colname] = new_lst

    return data_instances


def generate_newyorker_lm_text_dataset(
    newyorker_dataset: Union[datasets.Dataset, datasets.dataset_dict.DatasetDict],
    scene_colname_and_special_token: Tuple[str, str],
    uncanny_colname_and_special_token: Tuple[str, str],
    caption_colname_and_special_token: Tuple[str, str],
    end_of_caption_special_token: str,
    merge_colname: str = "text",
    batch_size: int = 4000,
    remove_cols: Optional[list] = None,
) -> Union[datasets.Dataset, datasets.dataset_dict.DatasetDict]:
    formatting_fn = partial(
        _merge_scene_uncanny_caption,
        scene_colname_and_special_token=scene_colname_and_special_token,
        uncanny_colname_and_special_token=uncanny_colname_and_special_token,
        caption_colname_and_special_token=caption_colname_and_special_token,
        end_of_caption_special_token=end_of_caption_special_token,
        merge_colname=merge_colname,
    )
    newyorker_dataset = newyorker_dataset.map(formatting_fn, batched=True, batch_size=batch_size).shuffle(seed=4740)
    if remove_cols is not None:
        newyorker_dataset = newyorker_dataset.remove_columns(remove_cols)
    return newyorker_dataset
