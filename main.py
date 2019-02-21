from model_select import MPModelSelector
import warnings
from configure import *
warnings.filterwarnings("ignore")


def main():
    model_selector = MPModelSelector(PIPELINE_OBJ_1, process_count=6)
    best = model_selector.optimize()
    print(best)


if __name__ == "__main__":
    main()
