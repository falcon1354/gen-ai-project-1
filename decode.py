# decode.py
# ---------
# After training, model should GENERATE questions.
#
# I made 2 ways:
#
# 1) Greedy decoding
#    At every step, pick the most likely next word.
#    Simple and fast.
#
# 2) Beam search
#    Keep a few best possible questions at the same time
#    (like top 4 paths), then choose the best final one.
#    Usually a bit better quality than greedy.

import torch
import sentencepiece as spm
import config


def load_sp():
    return spm.SentencePieceProcessor(model_file=config.SP_MODEL_FILE)


@torch.no_grad()
def greedy_decode(model, src, src_lens, max_len=None):
    """Generate questions using greedy method."""
    model.eval()

    if max_len is None:
        max_len = config.MAX_DECODE_LEN

    device = src.device
    batch_size = src.size(0)

    # first encode the input sentence
    encoder_outputs, hidden, cell = model.encoder(src, src_lens)
    mask = model.make_src_mask(src)

    # start with <s>
    input_token = torch.full((batch_size,), config.BOS_ID, dtype=torch.long, device=device)

    # track which examples already finished (got </s>)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # store generated token ids for each example
    results = []
    for i in range(batch_size):
        results.append([])

    for step in range(max_len):
        prediction, hidden, cell, _ = model.decoder.forward_step(
            input_token, hidden, cell, encoder_outputs, mask
        )

        # pick highest probability token
        next_token = prediction.argmax(1)

        for i in range(batch_size):
            if finished[i] == False:
                tok = next_token[i].item()
                if tok == config.EOS_ID:
                    finished[i] = True
                else:
                    results[i].append(tok)

        input_token = next_token

        # if everyone finished, stop early
        if finished.all():
            break

    return results

def trim_padding(src, src_lens):
    """
    When we take 1 example from a batch, src still has PAD tokens at the end.
    The encoder ignores PAD using src_lens, but attention mask must match
    encoder output length. So we cut off the PAD part here.
    """
    actual_len = int(src_lens[0].item())
    src = src[:, :actual_len]
    src_lens = torch.tensor([actual_len], dtype=torch.long, device=src.device)
    return src, src_lens

@torch.no_grad()
def beam_search_decode(model, src, src_lens, beam_size=None, max_len=None):
    """
    Beam search for 1 example only (batch size must be 1).

    Simple idea:
    - keep top beam_size partial questions
    - expand each one by top beam_size next words
    - keep best beam_size again
    - repeat
    """
    model.eval()

    if src.size(0) != 1:
        raise ValueError("beam_search_decode only works for one sentence at a time")

    if beam_size is None:
        beam_size = config.BEAM_SIZE
    if max_len is None:
        max_len = config.MAX_DECODE_LEN

    device = src.device

    # remove PAD tokens so mask size matches encoder output
    src, src_lens = trim_padding(src, src_lens)
    
    encoder_outputs, hidden, cell = model.encoder(src, src_lens)
    mask = model.make_src_mask(src)

    # each beam = [token_list, score, hidden, cell]
    beams = [[[], 0.0, hidden, cell]]
    completed = []

    for step in range(max_len):
        new_beams = []

        for tokens, score, h, c in beams:
            # what token do we feed now?
            if len(tokens) == 0:
                input_token = torch.tensor([config.BOS_ID], device=device)
            else:
                input_token = torch.tensor([tokens[-1]], device=device)

            prediction, new_h, new_c, _ = model.decoder.forward_step(
                input_token, h, c, encoder_outputs, mask
            )

            # log probabilities are better for summing scores
            log_probs = torch.log_softmax(prediction, dim=1).squeeze(0)
            top_values, top_indexes = torch.topk(log_probs, beam_size)

            for k in range(beam_size):
                log_p = top_values[k].item()
                tok = int(top_indexes[k].item())
                new_score = score + log_p

                if tok == config.EOS_ID:
                    # finished this candidate
                    # divide by length so longer questions are not always worse
                    if len(tokens) == 0:
                        norm = new_score
                    else:
                        norm = new_score / len(tokens)
                    completed.append([tokens, norm])
                else:
                    new_tokens = tokens + [tok]
                    new_beams.append([new_tokens, new_score, new_h, new_c])

        # keep only best partial beams
        # sort by score / length
        def score_key(item):
            toks = item[0]
            sc = item[1]
            if len(toks) == 0:
                return sc
            return sc / len(toks)

        new_beams = sorted(new_beams, key=score_key, reverse=True)
        beams = new_beams[:beam_size]

        if len(beams) == 0:
            break

    # also add unfinished beams
    for tokens, score, h, c in beams:
        if len(tokens) == 0:
            norm = score
        else:
            norm = score / len(tokens)
        completed.append([tokens, norm])

    if len(completed) == 0:
        return []

    # pick best completed sequence
    completed = sorted(completed, key=lambda x: x[1], reverse=True)
    best_tokens = completed[0][0]
    return best_tokens


def ids_to_text(sp, ids):
    """Convert token ids back to Urdu text."""
    clean = []
    for i in ids:
        if i == config.PAD_ID or i == config.BOS_ID or i == config.EOS_ID:
            continue
        clean.append(i)
    text = sp.decode(clean)
    return text


def generate_question(model, sp, source_text, device, beam_size=None):
    """
    Helper used by web app and predict.py
    Returns both greedy and beam outputs.
    """
    model.eval()

    # text -> ids
    src_ids = sp.encode(source_text, out_type=int)
    src_ids = src_ids[: config.MAX_SRC_LEN]

    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_lens = torch.tensor([len(src_ids)], dtype=torch.long, device=device)

    greedy_ids = greedy_decode(model, src, src_lens)[0]
    beam_ids = beam_search_decode(model, src, src_lens, beam_size=beam_size)

    result = {
        "greedy": ids_to_text(sp, greedy_ids),
        "beam": ids_to_text(sp, beam_ids),
    }
    return result
