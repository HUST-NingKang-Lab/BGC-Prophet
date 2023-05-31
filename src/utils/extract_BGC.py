#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import lmdb
import pickle
# import gc
import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

# torch.set_num_threads(128)
# gc.collect()
# torch.cuda.empty_cache()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "lmdb_path",
        type=pathlib.Path,
        help="path to save the lmdb file",
    )
    # parser.add_argument(
    #     "output_dir",
    #     type=pathlib.Path,
    #     help="output directory for extracted representations",
    # )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="specify which representations to return",
        required=True,
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="truncate sequences longer than the given value",
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def run(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches, num_workers=8
    )
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    args.lmdb_path.mkdir(parents=True, exist_ok=True)
    # args.output_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = str(args.lmdb_path)
    map_size = 307374182400
    env = lmdb.open(lmdb_path, subdir=True, map_size=map_size, readonly=False, meminit=False, map_async=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    results = []
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                # args.output_file = args.output_dir / f"{label}.pt"
                # args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(args.truncation_seq_length, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in args.include:
                    result["representations"] = {
                        layer: t[i, 1 : truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                if "mean" in args.include:
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in args.include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[i, : truncate_len, : truncate_len].clone()

                results.append(result)
            if batch_idx!=0 and batch_idx%1000 == 0:
                with env.begin(write=True) as txn:
                    for result_w in results:
                        label_w = result_w["label"]
                        txn.put(label_w.encode('ascii'), pickle.dumps(result_w))
                results = []
        if results:
            with env.begin(write=True) as txn:
                for result_w in results:
                    label_w = result_w["label"]
                    txn.put(label_w.encode('ascii'), pickle.dumps(result_w))


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()

