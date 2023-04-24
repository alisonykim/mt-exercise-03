import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os


ppl_types = ['train', 'val', 'test']
directory = '../ppls'


def create_ppl_table(ppl_type):
    '''Create a table with perplexities from all models (i.e. with for each dropout setting)'''
    ppl_df = pd.DataFrame()
    # Add perplexities to dataframe for each dropout setting
    for filename in glob.glob(directory + '/*' + ppl_type + '.csv'):
        dropout = filename.split('_')[1]
        df = pd.read_csv(filename)
        if ppl_type != 'test':
            if 'epoch' not in ppl_df.columns:
                ppl_df['Epoch'] = df['epoch']
            ppl_df[dropout + '% ' + 'Dropout'] = df.iloc[:, [1]]
        else:
            ppl_df[dropout + '% ' + 'Dropout'] = df.iloc[:, [0]]
    # reorder columns
    if ppl_type != 'test':
        cols = ["Epoch", "0% Dropout", "20% Dropout", "50% Dropout", "70% Dropout", "90% Dropout"]
        ppl_df = ppl_df[cols]
    else:
        cols = ["0% Dropout","20% Dropout","50% Dropout", "70% Dropout", "90% Dropout"]
        ppl_df = ppl_df[cols]
    return ppl_df


# Create tables for train, val and test perplexities each
for ppl_type in ppl_types:
    final_table = create_ppl_table(ppl_type)
    path = '../ppl_tables'
    if not os.path.exists(path):
        os.makedirs(path) 
    final_table.to_csv(path+'/'+ppl_type+'.csv', index=False)
    # create line charts for train and val tables
    if ppl_type != 'test':
        line_chart = sns.lineplot(x='Epoch', y='value', hue='variable', 
             data=pd.melt(final_table, ['Epoch']),
             palette=['red', 'blue', 'purple', 'pink', 'orange'])
        line_chart.set(xlabel='Epoch', ylabel='Perplexity')
        plt.show()
        plot_path = '../ppl_plot'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        line_chart.figure.savefig(plot_path+'/'+ppl_type+'_chart.png')