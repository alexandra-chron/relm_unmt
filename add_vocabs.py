import sys

if __name__ == '__main__':
    hmr_lang = sys.argv[1]
    lmr_lang = sys.argv[2]

    print(hmr_lang, lmr_lang)
    mono_path = './data/{}/'.format(hmr_lang)
    proc_path = './data/{}-{}/'.format(lmr_lang, hmr_lang)

    lmr = []
    hmr_vocab = []
    hmr_vocab_and_ind = []
    new_items_to_be_added = []
    overlapping_words = []
    with open(mono_path + 'vocab.' + hmr_lang, 'r') as file1:
        for line in file1:
            if len(line.split()) == 2:
                hmr_vocab.append(line.split()[0])
                hmr_vocab_and_ind.append(line)
        # print("Length of hmr vocabulary is {}".format(len(hmr_vocab)))

    with open(proc_path + 'vocab.' + lmr_lang, 'r') as file2:
        for line in file2:
            lmr.append(line.split()[0])
            if len(line.split()) == 2:
                if not line.split()[0] in hmr_vocab:
                    new_items_to_be_added.append(line)
                if line.split()[0] in hmr_vocab:
                    overlapping_words.append(line)
        # print("Length of lmr language is {}\n".format(len(lmr)))
    biggest_lmr = new_items_to_be_added[0].split()[1]
    # print("Length of new items to be added is {} and the final vocabulary will "
    #       "have {} items\n".format(len(new_items_to_be_added),
    #                              int(len(new_items_to_be_added)) + int(len(hmr_vocab))))

    intersection = set(lmr).intersection(hmr_vocab)
    # print("Length of intersection of 2 vocabs is {}\n".format(len(intersection)))

    final_vocab = []

    for line in hmr_vocab_and_ind:
        value = int(line.split()[1]) + int(biggest_lmr)
        final_vocab.append("{} {}\n".format(line.split()[0], value))

    for line in new_items_to_be_added:
        final_vocab.append(line)

    with open(proc_path + 'vocab.{}-{}-ext-by-{}'.format(lmr_lang, hmr_lang, len(new_items_to_be_added)), 'w') as f:
        for item in final_vocab:
            f.write("%s" % item)

    sys.stdout.write('vocab.%s-%s-ext-by-%s' % (lmr_lang, hmr_lang, len(new_items_to_be_added)))
