import torch.nn as nn

class SentenceLengthBiasNet(nn.Module):

    def __init__(self):
        super(SentenceLengthBiasNet, self).__init__()
        # self.ff1 = nn.Linear(1, 1)
        # self.relu1 = nn.Tanh()
        # self.ff2 = nn.Linear(1, 1)
        # self.relu2 = nn.Tanh()
        self.out = nn.Linear(1, 1)

    def forward(self, x):
        # x = self.relu1(self.ff1(x))
        # x = self.relu2(self.ff2(x))
        x = self.out(x)

        return x

# Utility function to get the sentence length (word count)

def get_sentence_length(df, source_or_target='target'):
    '''This method takes dataframes as input and assumes that the text columns
    are named "text_a and "text_b. '''

    df_out = df.copy()
    if source_or_target == 'target':
        df_out['sentence_length'] = df_out['text_b'].str.split().str.len()

    else:
        df_out['sentence_length'] = df_out['text_a'].str.split().str.len()

    return df_out