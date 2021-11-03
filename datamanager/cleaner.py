import pandas as pd
import re


class DataCleaner:
    def __init__(self, filename, cols):
        self.raw_df = pd.read_excel(filename)
        self.cols = cols

    @staticmethod
    def _get_english(abstract: str) -> str:
        exp = re.compile("\[EN(.+?)(\[|$)", flags=re.DOTALL)
        match = exp.search(abstract)
        if match is not None:
            return match.group(1)

    @staticmethod
    def _filter_chars(abstract: str) -> str:
        if abstract is not None:
            return re.sub(r"[^a-z A-Z]+", "", abstract)

    def clean(self):
        if self.cols is not None:
            self.raw_df = self.raw_df[self.cols]

        if "Abstract" in self.raw_df.columns:
            self.raw_df.Abstract = self.raw_df.apply(
                lambda row: self._get_english(row["Abstract"]), axis=1
            )
            self.raw_df.Abstract = self.raw_df.apply(
                lambda row: self._filter_chars(row["Abstract"]), axis=1
            )

        if "Title" in self.raw_df.columns:
            self.raw_df.Title = self.raw_df.apply(
                lambda row: self._get_english(row["Title"]), axis=1
            )
            self.raw_df.Title = self.raw_df.apply(
                lambda row: self._filter_chars(row["Title"]), axis=1
            )

        self.raw_df.dropna(inplace=True)
        self.raw_df.reset_index(drop=True, inplace=True)

    def save_abstract_to_txt(self, relevant: bool, save_loc: str) -> str:
        if "Abstract" not in self.raw_df.columns:
            return
        else:
            if relevant:
                cat = "relevant"
            else:
                cat = "irrelevant"

            for i, row in self.raw_df.iterrows():
                abstract = row["Abstract"]

                with open(
                    f"{save_loc}/{cat}/{cat}_{i}.txt", "w"
                ) as text_file:
                    text_file.write(abstract)


if __name__ == "__main__":
    # Only for testing, depends on local environment
    relevant = "/Users/august/PycharmProjects/Patent-Classifier/datamanager/raw-data/sp_relevant.xlsx"
    irrelevant = "/Users/august/PycharmProjects/Patent-Classifier/datamanager/raw-data/sp_irrelevant.xlsx"

    relevant_cleaner = DataCleaner(
        filename=relevant,
        cols=["Abstract", "Title", "International Patent Classification (IPC)"],
    )
    irrelevant_cleaner = DataCleaner(
        filename=irrelevant,
        cols=["Abstract", "Title", "International Patent Classification (IPC)"],
    )

    relevant_cleaner.clean()
    irrelevant_cleaner.clean()

    relevant_cleaner.save_abstract_to_txt(
        relevant=True,
        save_loc="/Users/august/PycharmProjects/Patent-Classifier/datasets/sp",
    )
    irrelevant_cleaner.save_abstract_to_txt(
        relevant=False,
        save_loc="/Users/august/PycharmProjects/Patent-Classifier/datasets/sp",
    )
