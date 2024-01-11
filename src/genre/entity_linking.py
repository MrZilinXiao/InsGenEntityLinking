# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch

from genre.trie import DummyTrieEntity, DummyTrieMention, Trie


def get_end_to_end_prefix_allowed_tokens_fn_hf_bart(
    tokenizer,
    sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: tokenizer.encode(x),
        lambda x: tokenizer.decode(torch.tensor(x)),
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        len(tokenizer) - 1,
        sentences,
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        mention_trie,
        candidates_trie,
        mention_to_candidates_dict,
    )


def get_end_to_end_prefix_allowed_tokens_fn_hf(  # llama used
    tokenizer,
    sentences: List[str],
    prompt_sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
    debug=False,
    # error_spacing=False
):
    return _get_end_to_end_prefix_allowed_tokens_fn_hf(
        lambda x: tokenizer.encode(x),
        lambda x: tokenizer.decode(torch.tensor(x)),
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        len(tokenizer) - 1,
        sentences,
        prompt_sentences,  # updated on 03/29
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        mention_trie,
        candidates_trie,
        mention_to_candidates_dict,
        debug
    )


def _get_end_to_end_prefix_allowed_tokens_fn_hf(  # llama used
    encode_fn,
    decode_fn,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    vocabulary_length,
    sentences: List[str],  # sentences to be linked
    prompt_sentences: List[str],  # sentences to be used as instructions
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
    debug=False
):

    assert not (
        candidates_trie is not None and mention_to_candidates_dict is not None
    ), "`candidates_trie` and `mention_to_candidates_dict` cannot be both != `None`"

    # save 4 special tokens for mentions and entity
    codes = {
        # n: encode_fn(" {}".format(c))[1]
        n: encode_fn("{}".format(c))[1]  # no space in llama when encoding special tokens
        for n, c in zip(
            (
                "start_mention_token",
                "end_mention_token",
                "start_entity_token",
                "end_entity_token",
            ),
            (
                start_mention_token,
                end_mention_token,
                start_entity_token,
                end_entity_token,
            ),
        )
    }
    codes["EOS"] = eos_token_id

    if debug:
        print(f"codes prepared: {codes}")

    if mention_trie is None:
        # no constraint on mentions
        mention_trie = DummyTrieMention(  # dummy trie returns a sequence of all tokens except bos + pad -> no constraint on mentions
            # **however pad_token_id is identical to eos_token_id**
            [
                i
                for i in range(vocabulary_length)
                if i
                not in (
                    bos_token_id,
                    pad_token_id,
                )
            ]
        )

    if candidates_trie is None and mention_to_candidates_dict is None:
        # no constraint on candidates; we do not use this
        candidates_trie = DummyTrieEntity(
            [
                i
                for i in range(vocabulary_length)
                if i
                not in (
                    bos_token_id,
                    pad_token_id,
                )
            ],
            codes,
        )

    # each sentence replace the bos into eos
    sent_origs = [[codes["EOS"]] + encode_fn(sent)[1:] for sent in sentences]  # this do not include instructions
    # this encode_fn has been manually appended EOS to sentences

    # record the length of each sentence's prompt part; prompt = instruction + input
    sent_prompt_lens = [len(encode_fn(sent)) for sent in prompt_sentences]

    if debug:
        print(f"sent_origs prepared: {sent_origs}")
        print(f"sent_prompt_lens prepared: {sent_prompt_lens}")

    def prefix_allowed_tokens_fn(batch_id, sent):  # receive a batch of sentences from hf

        sent = sent.tolist()  # **a tokenized generated sentence**; consists of prompt parts!

        # if debug:
        #     print(f"already generated sent ids: {sent}, len: {len(sent)}\nalready generated sent: {decode_fn(sent)}")

        sent = [eos_token_id] + sent[sent_prompt_lens[batch_id]:]  # remove prompt parts while keep eos

        if debug:
            print(f"cropped generated sentence: {sent}\ncropped generated sentence: {decode_fn(sent)}")

        status = get_status(sent)  # decide current status

        if debug:
            print(f"current status: {status}")

        sent_orig = sent_origs[batch_id]  # for generation guidance

        if debug:
            print(f"original sentence: {sent_orig}")
        # sent_orig = [codes["EOS"]] + sent_origs[batch_id][sent_prompt_lens[batch_id]: ]  # remove prompt parts, with eos prepended

        if status == "o":
            # passed, will not return int
            trie_out = get_trie_outside(sent, sent_orig)
        elif status == "m":
            # passed, will not return int
            trie_out = get_trie_mention(sent, sent_orig)  # if trie_out -> [], will predict on `0`
        elif status == "e":
            # possibly return an int
            trie_out = get_trie_entity(sent, sent_orig)
            if trie_out == codes["EOS"]:  # that special </s> will trigger outside generation
                print(f"CAUTION: trie_out is EOS, so we are outside!")
                trie_out = get_trie_outside(sent, sent_orig)
        else:
            raise RuntimeError

        if isinstance(trie_out, int):
            trie_out = [trie_out]
            print(f"WARNING: trie_out is int: {trie_out}")
        if debug:
            print(f"trie_out: {trie_out}, ", end='')
            print(f"which means allowed tokens are: {[decode_fn(trie_out_single) for trie_out_single in trie_out]}")
            print('------------------------')

        return trie_out  # all allowed generated position

    def get_status(sent):
        c = [
            codes[e]
            for e in (
                "start_mention_token",
                "end_mention_token",
                "start_entity_token",
                "end_entity_token",
            )
        ]
        status = sum(e in c for e in sent) % 4
        # 0 -> outside any mention or entity (passed through 4x boundary tokens);
        # 1 -> mention;
        # 2 -> entity about to begin;
        # 3 -> in entity;

        if status == 0:
            return "o"
        elif status == 1:
            return "m"
        else:
            return "e"

    def get_trie_outside(sent, sent_orig):
        pointer_end = get_pointer_end(sent, sent_orig)  # get j from reference sentence
        if debug:
            print(f'trie_outside sent: {sent}')
            print(f'trie_outside sent_orig: {sent_orig}')
            print(f"pointer_end: {pointer_end}")

        if pointer_end:
            if sent_orig[pointer_end] != codes["EOS"] and sent_orig[
                pointer_end
            ] in mention_trie.get([]):
                return [sent_orig[pointer_end], codes["start_mention_token"]]
                # when 1. ref sent not ending, 2. current ref token in mention allow list -> allow generating a mention start
            else:
                return [sent_orig[pointer_end]]  # otherwise, only allow identical generation
        else:
            return []

    def get_pointer_end(sent, sent_orig):
        # return the index of the last token_id in the generated sentence that matches sent_orig
        i = 0
        j = 0
        while i < len(sent):
            if sent[i] == sent_orig[j]:
                i += 1
                j += 1
            elif (
                sent[i] == codes["start_mention_token"]
                or sent[i] == codes["end_mention_token"]
            ):  # skip mention boundary tokens
                i += 1
            elif sent[i] == codes["start_entity_token"]:  # skip entity identifier in generated sentence
                i += 1
                while sent[i] != codes["end_entity_token"]:
                    i += 1
                i += 1
            else:
                return None

        return j if j != len(sent_orig) else None

    def get_trie_mention(sent, sent_orig):
        if debug:
            print(f'trie_mention sent: {sent}')
            print(f'trie_mention sent_orig: {sent_orig}')
        pointer_start, _ = get_pointer_mention(sent)  # find mention start and end in generated sentence
        if pointer_start + 1 < len(sent):  # if mention start is not just generated
            ment_next = mention_trie.get(sent[pointer_start + 1 :])  
            # put already generated mention seq in, get next possible tokens using trie
        else:
            ment_next = mention_trie.get([])  # the mention is just generated, you can pick any next token in possible mention start tokens
            # however, do not allow immediate close this mention

        pointer_end = get_pointer_end(sent, sent_orig)  
        # return the index of the last token_id in the generated sentence that matches sent_orig

        if pointer_end:
            if sent_orig[pointer_end] != codes["EOS"]:
                if sent_orig[pointer_end] in ment_next:
                    if codes["EOS"] in ment_next: # if generated mention is valid, can continue generate mention span or close mention
                        return [sent_orig[pointer_end], codes["end_mention_token"]]  
                    else:  # incomplete mention, can only continue generate next word
                        return [sent_orig[pointer_end]]
                # in the middle of some words, can only generate the same word; we currently do not implement this
                
                
                elif codes["EOS"] in ment_next:  
                    # can not continue generate mention span (as sent_orig[pointer_end] not in ment_next)
                    # -> only to close the mention
                    return [codes["end_mention_token"]]
                else:
                    return []
            else:  # when sent_orig pointer already hits EOS, we can only generate end_mention_token
                return [codes["end_mention_token"]]
        else:
            return []

    def get_pointer_mention(sent):
        pointer_end = -1
        for i, e in enumerate(sent):
            if e == codes["start_mention_token"]:
                pointer_start = i
            elif e == codes["end_mention_token"]:
                pointer_end = i

        return pointer_start, pointer_end

    def get_trie_entity(sent, sent_orig):
        pointer_start, pointer_end = get_pointer_mention(sent)  # do not care sent_orig

        if pointer_start + 1 != pointer_end:  # valid mention span
            mention = decode_fn(sent[pointer_start + 1 : pointer_end]).strip()  # mention_str
            if debug:
                print(f"mention string: {mention}")

            if candidates_trie is not None:
                candidates_trie_tmp = candidates_trie
            elif mention_to_candidates_dict is not None:  # dynamic construct trie of all candidates when needed
                if debug:
                    print(f"candidates of this mention: {mention_to_candidates_dict.get(mention, ['NIL'])}")
                candidates_trie_tmp = Trie(
                    [
                        encode_fn(
                            # "{} {} {} {}</s>".format(  # add EOS manually
                            "{} {} {} {}</s>".format(
                                end_mention_token,
                                start_entity_token,
                                e,
                                end_entity_token,
                            )
                        )[1:]
                        for e in mention_to_candidates_dict.get(mention, ["NIL"])
                    ]
                )
            else:
                raise RuntimeError()

            return candidates_trie_tmp.get(sent[pointer_end:])

        return []

    return prefix_allowed_tokens_fn


def get_end_to_end_prefix_allowed_tokens_fn_fairseq(
    model,
    sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: model.encode(x).tolist(),
        lambda x: model.decode(torch.tensor(x)),
        model.model.decoder.dictionary.bos(),
        model.model.decoder.dictionary.pad(),
        model.model.decoder.dictionary.eos(),
        len(model.model.decoder.dictionary),
        sentences,
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        mention_trie,
        candidates_trie,
        mention_to_candidates_dict,
    )


def _get_end_to_end_prefix_allowed_tokens_fn(  # for original bart
    encode_fn,
    decode_fn,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    vocabulary_length,
    sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
):

    assert not (
        candidates_trie is not None and mention_to_candidates_dict is not None
    ), "`candidates_trie` and `mention_to_candidates_dict` cannot be both != `None`"

    # save 4 special tokens for mentions and entity
    codes = {
        n: encode_fn(" {}".format(c))[1]
        for n, c in zip(
            (
                "start_mention_token",
                "end_mention_token",
                "start_entity_token",
                "end_entity_token",
            ),
            (
                start_mention_token,
                end_mention_token,
                start_entity_token,
                end_entity_token,
            ),
        )
    }
    codes["EOS"] = eos_token_id

    print(f"codes: {codes}")

    if mention_trie is None:
        # no constraint on mentions
        mention_trie = DummyTrieMention(
            [
                i
                for i in range(vocabulary_length)
                if i
                not in (
                    bos_token_id,
                    pad_token_id,
                )
            ]
        )

    if candidates_trie is None and mention_to_candidates_dict is None:
        # no constraint on candidates
        candidates_trie = DummyTrieEntity(
            [
                i
                for i in range(vocabulary_length)
                if i
                not in (
                    bos_token_id,
                    pad_token_id,
                )
            ],
            codes,
        )

    sent_origs = [[codes["EOS"]] + encode_fn(sent)[1:] for sent in sentences]

    def prefix_allowed_tokens_fn(batch_id, sent):  # receive a batch of sentences from hf

        sent = sent.tolist()

        print(f"already generated sentence: {sent}, len: {len(sent)}")

        status = get_status(sent)

        print(f"current status: {status}")

        sent_orig = sent_origs[batch_id]

        print(f"sent_orig: {sent_orig}")

        if status == "o":
            trie_out = get_trie_outside(sent, sent_orig)
        elif status == "m":
            trie_out = get_trie_mention(sent, sent_orig)
        elif status == "e":
            trie_out = get_trie_entity(sent, sent_orig)
            if trie_out == codes["EOS"]:
                trie_out = get_trie_outside(sent, sent_orig)
        else:
            raise RuntimeError

        print(f"trie_out: {trie_out}, which means allowed tokens are: {[decode_fn(trie_out_single) for trie_out_single in trie_out]}")

        print('------------------------')

        return trie_out

    def get_status(sent):
        c = [
            codes[e]
            for e in (
                "start_mention_token",
                "end_mention_token",
                "start_entity_token",
                "end_entity_token",
            )
        ]
        status = sum(e in c for e in sent) % 4

        if status == 0:
            return "o"
        elif status == 1:
            return "m"
        else:
            return "e"

    def get_trie_outside(sent, sent_orig):
        print(f'trie_outside sent: {sent}')
        print(f'trie_outside sent_orig: {sent_orig}')
        pointer_end = get_pointer_end(sent, sent_orig)
        print(f"pointer_end: {pointer_end}")
        if pointer_end:
            if sent_orig[pointer_end] != codes["EOS"] and sent_orig[
                pointer_end
            ] in mention_trie.get([]):
                return [sent_orig[pointer_end], codes["start_mention_token"]]
            else:
                return [sent_orig[pointer_end]]
        else:
            return []

    def get_pointer_end(sent, sent_orig):
        i = 0
        j = 0
        while i < len(sent):
            if sent[i] == sent_orig[j]:
                i += 1
                j += 1
            elif (
                sent[i] == codes["start_mention_token"]
                or sent[i] == codes["end_mention_token"]
            ):
                i += 1
            elif sent[i] == codes["start_entity_token"]:
                i += 1
                while sent[i] != codes["end_entity_token"]:
                    i += 1
                i += 1
            else:
                return None

        return j if j != len(sent_orig) else None

    def get_trie_mention(sent, sent_orig):

        pointer_start, _ = get_pointer_mention(sent)
        if pointer_start + 1 < len(sent):
            ment_next = mention_trie.get(sent[pointer_start + 1 :])
        else:
            ment_next = mention_trie.get([])

        pointer_end = get_pointer_end(sent, sent_orig)

        if pointer_end:
            if sent_orig[pointer_end] != codes["EOS"]:
                if sent_orig[pointer_end] in ment_next:
                    if codes["EOS"] in ment_next:
                        return [sent_orig[pointer_end], codes["end_mention_token"]]
                    else:
                        return [sent_orig[pointer_end]]
                elif codes["EOS"] in ment_next:
                    return [codes["end_mention_token"]]
                else:
                    return []
            else:
                return [codes["end_mention_token"]]
        else:
            return []

    def get_pointer_mention(sent):
        pointer_end = -1
        for i, e in enumerate(sent):
            if e == codes["start_mention_token"]:
                pointer_start = i
            elif e == codes["end_mention_token"]:
                pointer_end = i

        return pointer_start, pointer_end

    def get_trie_entity(sent, sent_orig):
        pointer_start, pointer_end = get_pointer_mention(sent)

        if pointer_start + 1 != pointer_end:
            mention = decode_fn(sent[pointer_start + 1 : pointer_end]).strip()
            print(f"mention string: {mention}")
            if candidates_trie is not None:
                candidates_trie_tmp = candidates_trie
            elif mention_to_candidates_dict is not None:  # dynamic construct trie of all candidates when needed
                print(f"candidates of this mention: {mention_to_candidates_dict.get(mention, ['NIL'])}")
                candidates_trie_tmp = Trie(
                    [
                        encode_fn(
                            " {} {} {} {}".format(
                                end_mention_token,
                                start_entity_token,
                                e,
                                end_entity_token,
                            )
                        )[1:]
                        for e in mention_to_candidates_dict.get(mention, ["NIL"])
                    ]
                )
            else:
                raise RuntimeError()

            return candidates_trie_tmp.get(sent[pointer_end:])

        return []

    return prefix_allowed_tokens_fn
