import logomaker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def activation_pwm(fmap, X, threshold=0.5, window=20, pad=False):
    # extract sequences with aligned activation
    window_left = int(window/2)
    window_right = window - window_left

    N,L,A = X.shape
    num_filters = fmap.shape[-1]

    W = []
    for filter_index in range(num_filters):

        # find regions above threshold
        coords = np.where(fmap[:,:,filter_index] > np.max(fmap[:,:,filter_index])*threshold)

        if len(coords) > 1:
            x, y = coords

            # sort score
            index = np.argsort(fmap[x,y,filter_index])[::-1]
            data_index = x[index].astype(int)
            pos_index = y[index].astype(int)

            # make a sequence alignment centered about each activation (above threshold)
            seq_align = []
            for i in range(len(pos_index)):

                if pad:
                    # determine position of window about each filter activation
                    start_window = pos_index[i] - window_left
                    end_window = pos_index[i] + window_right

                    # check to make sure positions are valid
                    if (start_window > 0) & (end_window < L):
                        seq = X[data_index[i], start_window:end_window, :]
                else:
                    seq = X[data_index[i], pos_index[i]:pos_index[i] + window, :]

                if seq.shape[0] == window:
                    seq_align.append(seq)

            # calculate position probability matrix
            if len(seq_align) > 1:#try:
                W.append(np.mean(seq_align, axis=0))
            else:
                W.append(np.ones((window,4))/4)
        else:
            W.append(np.ones((window,4))/4)

    return np.array(W)


def plot_filters(W, num_cols=8, figsize=(30, 10)):
	num_filters = len(W)
	num_widths = int(np.ceil(num_filters / num_cols))

	fig = plt.figure(figsize=figsize)
	fig.subplots_adjust(hspace=0.1, wspace=0.1)

	logos = []
	for n, w in enumerate(W):
		ax = fig.add_subplot(num_widths, num_cols, n + 1)

		# calculate sequence logo heights
		I = np.log2(4) + np.sum(w * np.log2(w + 1e-10), axis=1, keepdims=True)
		logo = np.maximum(I * w, 1e-7)

		L, A = w.shape
		counts_df = pd.DataFrame(data=0.0, columns=list('ACGT'), index=list(range(L)))
		for a in range(A):
			for l in range(L):
				counts_df.iloc[l, a] = logo[l, a]

		logomaker.Logo(counts_df, ax=ax)
		ax = plt.gca()
		ax.set_ylim(0, 2)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('none')
		ax.xaxis.set_ticks_position('none')
		plt.xticks([])
		plt.yticks([])

		logos.append(logo)

	return fig, logos


def clip_filters(W, threshold=0.5, pad=3):
    W_clipped = []
    for w in W:
        L,A = w.shape
        entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=1)
        index = np.where(entropy > threshold)[0]
        if index.any():
            start = np.maximum(np.min(index)-pad, 0)
            end = np.minimum(np.max(index)+pad+1, L)
            W_clipped.append(w[start:end,:])
        else:
            W_clipped.append(w)

    return W_clipped


def meme_generate(W, output_file='meme.txt', prefix='filter'):
    # background frequency
    nt_freqs = [1./4 for i in range(4)]

    # open file for writing
    f = open(output_file, 'w')

    # print intro material
    f.write('MEME version 4\n')
    f.write('\n')
    f.write('ALPHABET= ACGT\n')
    f.write('\n')
    f.write('Background letter frequencies:\n')
    f.write('A %.4f C %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
    f.write('\n')

    for j, pwm in enumerate(W):
        L, A = pwm.shape
        f.write('MOTIF %s%d \n' % (prefix, j))
        f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (L, L))
        for i in range(L):
            f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[i,:]))
        f.write('\n')

    f.close()