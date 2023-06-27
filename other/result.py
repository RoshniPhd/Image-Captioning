import pandas as pd
from colorama import Fore, init
import matplotlib.pyplot as plt

init(autoreset=True)


def result():
    colum = ['Faster R-CNN [40]', 'gLSTM [34]', 'LSTM', 'DBN', 'BI-GRU', 'CNN+CMBO', 'CNN+SSA', 'CNN+WHO', 'CNN+SSO','CNN','CNN + SMO-SCME']
    column = ['Faster R-CNN [40]', 'gLSTM [34]', 'LSTM', 'DBN', 'BI-GRU', 'CNN+CMBO', 'CNN+SSA', 'CNN+WHO', 'CNN+SSO','LSTM[41]','CNN','CNN + SMO-SCME']

    plot_result = pd.read_csv(f'pre_evaluated/saved/Optimization.csv', index_col=[0, 1])
    plot_result.columns = column

    conv = pd.read_csv('pre_evaluated/saved/convergence 60.csv')
    conv.plot(xlabel='Iteration', ylabel='Cost Function')
    plt.savefig('result/convergence.png')

    indx = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'CIDER', 'ROUGE', 'METEOR']

    for i in range(60, 91, 10):
        avg = plot_result.loc[i, :]
        avg.reset_index(drop=True, level=0)
        avg.to_csv(f'result/' + str(i) + '.csv')
        print('\n\t', Fore.LIGHTBLUE_EX + str(i))
        print(avg.to_markdown())

    print('\n\t', Fore.LIGHTBLUE_EX + 'Statistical Analysis')
    print(pd.read_csv(f'pre_evaluated/saved/statistics analysis.csv', header=0, names=colum).to_markdown())

    for idx, jj in enumerate(indx):
        new_ = plot_result.loc[([60, 70, 80, 90], [jj]), :]
        new_.reset_index(drop=True, level=1, inplace=True)
        new_.plot(figsize=(10, 6), kind='bar', width=0.8, use_index=True,
                  xlabel='Learning Percentage', ylabel=jj.upper(), rot=0)
        plt.subplots_adjust(bottom=0.2)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=5)
        plt.savefig('result/' + jj + '.png')
        plt.show(block=False)

    plt.show()

