from model_select import MPModelSelector
import warnings
warnings.filterwarnings("ignore")


def main():
    model_selector = MPModelSelector(q_size=4)
    model_selector.optimize()

if __name__ == "__main__":
    main()