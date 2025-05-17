import os
from result_writer import ResultWriter
from sentence_reader import SentenceReader
from main import write_ensembled_predictions_conll_format, process_final_submission_output
from config import Config
from logger import logger


def main():
    config = Config('config.yaml')
    result_writer = ResultWriter(
        adapted_model_path=config.adapted_model_path,
        work_dir=config.work_dir
    )

    fold_dirs_to_predict = result_writer.discover_fold_work_dirs(
        main_work_dir=config.work_dir)

    if fold_dirs_to_predict:
        result_writer.run_prediction_and_ensembling_pipeline(
            fold_work_dirs=fold_dirs_to_predict)
    else:
        logger.warning(
            "No fold directories found or specified. Skipping prediction and ensembling.")


if __name__ == "__main__":
    main()
