# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True,
        lprobs_old=None, lprobs_mle=None, config=None, sample=None):
    from fairseq_cli.train import model_old, model_mle

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    batch_size = sample['target'].shape[0]

    # importance sampling; theta below is the same thing as pi in the paper 
    if lprobs_old is None or lprobs_mle is None:
        weight_theta_hat = 1.0  # the weight correspond to a slightly old version of pi
    else:
        weight_theta_hat = lprobs_old.gather(dim=-1, index=target)
        weight_mle = lprobs_mle.gather(dim=-1, index=target)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
        if lprobs_old is not None or lprobs_mle is not None:
            weight_theta_hat.masked_fill_(pad_mask, 1.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        nll_loss_old = nll_loss_old.squeeze(-1)
        raise NotImplementedError

    if lprobs_old is not None or lprobs_mle is not None:
        with torch.no_grad():
            # the below code hardcodes for now
            if config.suffix_num > 0:
                def obtain_suffix_weights_kk(weight_fn, kk):
                    fn_weight_original = weight_fn.clone()
                    if kk == 0: 
                        fn_weight_nextk = fn_weight_original
                    else: 
                        fn_weight_nextk = fn_weight_original.clone()
                        fn_weight_nextk = fn_weight_nextk.reshape(batch_size, -1)
                        fn_weight_original = fn_weight_original.reshape(batch_size, -1)

                        fn_weight_nextk[:, :-kk] = fn_weight_original[:, kk:].clone()
                        for aa in range(1,kk+1):
                            if aa <= fn_weight_nextk.shape[1]:
                                fn_weight_nextk[:, -aa].fill_(1.0)
                    if config.reward_type == 'sump':
                        fn_weight_nextk = fn_weight_nextk
                    elif config.reward_type == 'logp':
                        fn_weight_nextk = torch.log(fn_weight_nextk+1e-10) - config.q_baseline

                    fn_weight_nextk = torch.clamp(fn_weight_nextk, min=config.trunc_min)  # warning
                    return fn_weight_nextk.reshape(-1, 1)

                if config.suffix_num == 5:
                    try:
                        weight_suffix = obtain_suffix_weights_kk(weight_mle, 0) + \
                            (config.gamma ** 1) * obtain_suffix_weights_kk(weight_mle, 1) + \
                            (config.gamma ** 2) * obtain_suffix_weights_kk(weight_mle, 2) + \
                            (config.gamma ** 3) * obtain_suffix_weights_kk(weight_mle, 3) + \
                            (config.gamma ** 4) * obtain_suffix_weights_kk(weight_mle, 4) + \
                            (config.gamma ** 5) * obtain_suffix_weights_kk(weight_mle, 5)
                    except:  # check sequence length
                        weight_suffix = obtain_suffix_weights_kk(weight_mle, 0) + \
                            (config.gamma ** 1) * obtain_suffix_weights_kk(weight_mle, 1) + \
                            (config.gamma ** 2) * obtain_suffix_weights_kk(weight_mle, 2) + \
                            (config.gamma ** 3) * obtain_suffix_weights_kk(weight_mle, 3)                     
                else:
                    # Can implement much more elegantly for longer suffix_num!
                    raise NotImplementedError(config.suffix_num)

                b1 = torch.clamp(weight_theta_hat, min=config.iw_min, max=1.0)  # warning
                b2 = weight_suffix
    else: 
        b1 = 1.0
        b2 = 1.0

    nll_loss_new = (b1 * b2) * nll_loss
    # Can also adjust smooth loss accordingly; but no big impact
    if reduce:
        nll_loss = nll_loss.sum()
        nll_loss_new = nll_loss_new.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss_new + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        from fairseq_cli.train import model_old, model_mle
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model_old.eval()
        model_mle.eval()
        with torch.no_grad():
            net_output_old = model_old(**sample['net_input'])
            net_output_mle = model_mle(**sample['net_input'])
        net_output = model(**sample['net_input'])

        if model.args.use_is_obj:
            loss, nll_loss = self.compute_loss(
                model, net_output, sample, reduce=reduce,
                output_old=net_output_old, output_mle=net_output_mle,
            ) 
        else:
            loss, nll_loss = self.compute_loss(
                model, net_output, sample, reduce=reduce, 
                output_old=None, output_mle=net_output_mle,
            ) 
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, output_old=None, output_mle=None):
        from fairseq_cli.train import model_old, model_mle

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)

        with torch.no_grad():
            if output_old is not None:
                lprobs_old = model_old.get_normalized_probs(output_old, log_probs=False)
                lprobs_old = lprobs_old.view(-1, lprobs.size(-1))
                lprobs_mle = model_mle.get_normalized_probs(output_mle, log_probs=False)
                lprobs_mle = lprobs_mle.view(-1, lprobs.size(-1))
            else:
                lprobs_old = None
                lprobs_mle = None

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce, 
            lprobs_old=lprobs_old, lprobs_mle=lprobs_mle, config=model.args, sample=sample,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
