import torch
from fairseq.models.bart import BARTModel

bart = BARTModel.from_pretrained('/path/to/model/directory/', checkpoint_file='model-name.pt', data_name_or_path='squad-bin')
bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32
with open('squad/test.source') as source, open('squad/test.hypo', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=1, lenpen=1.0, max_len_b=35, min_len=5, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []
        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=1, lenpen=1.0, max_len_b=35, min_len=5, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
