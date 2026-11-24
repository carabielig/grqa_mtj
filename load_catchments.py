from scripts import utils

if __name__ == "__main__":
    utils.cut_and_save_dtm_and_slp(
        "/p/tmp/carabi/data/dtm",
        "/p/tmp/carabi/data/slp",
        "/p/tmp/carabi/data/mask",
        "/p/tmp/carabi/data/dtm_lir",
        "/p/tmp/carabi/data/slp_lir",
    )
