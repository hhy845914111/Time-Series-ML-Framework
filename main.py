from model_select import MPModelSelector
import warnings
warnings.filterwarnings("ignore")


def main():
    model_selector = MPModelSelector(process_count=15, q_size=8)
    best = model_selector.optimize()
    print(best)


if __name__ == "__main__":
    main()
