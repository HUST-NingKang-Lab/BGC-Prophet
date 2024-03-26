#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import lmdb
import pickle
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from .baseCommand import baseCommand

class ExtractCommand(baseCommand):
    name = "extract"
    description = "Extract all tokens' mean representations and save to a lmdb file for sequences in a FASTA file"

    def add_arguments(self, parser):
        parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
        )
        parser.add_argument(
            "fasta",
            type=pathlib.Path,
            help="FASTA source on which to extract representations",
        )
        parser.add_argument(
            "lmdb_path",
            type=pathlib.Path,
            default=pathlib.Path("./output/lmdb/"),
            help="path to save the lmdb file",
        )
        # parser.add_argument(
        #     "output_dir",
        #     type=pathlib.Path,
        #     help="output directory for extracted representations",
        # )

        parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
        parser.add_argument("--directory", '-d', action='store_true', required=False, default=False, help='indicate the input is a directory')
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
        parser.add_argument(
            "--write_batches",
            type=int,
            default=1000,
            help="write to lmdb every this many batches, RAM usage increases with this number",
        )

        parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")

    def handle(self, args):
        run(args)



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

    def extract_and_save(fasta_file, model, alphabet, args):
        args.lmdb_path.mkdir(parents=True, exist_ok=True)
        fasta_seqs = list(SeqIO.parse(fasta_file, "fasta"))
        fasta_id_seqs = [SeqRecord(Seq(str(seq.seq)), id=seq.id, description="") for seq in fasta_seqs]
        temp_file = args.lmdb_path / ('temp_'+str(fasta_file.name))
        SeqIO.write(fasta_id_seqs, temp_file, "fasta")
        dataset = FastaBatchedDataset.from_file(temp_file)
        os.remove(temp_file)
        batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches, num_workers=8
        )
        print(f"Read {fasta_file} with {len(dataset)} sequences")

        # args.output_dir.mkdir(parents=True, exist_ok=True)
        map_size = 307374182400
        env = lmdb.open(str(args.lmdb_path), subdir=True, map_size=map_size, readonly=False, meminit=False, map_async=True)
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

                if batch_idx!=0 and batch_idx%args.write_batches == 0:
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


    if args.directory:
        if not args.fasta.is_dir():
            raise ValueError("The input is not a directory.")
        else:
            fasta_files = [f for f in args.fasta.iterdir() if f.is_file() and f.suffix in ['.fasta', '.faa']]
            for fasta_file in fasta_files:
                extract_and_save(fasta_file=fasta_file, model=model, alphabet=alphabet, args=args)
    else:
        if args.fasta.is_dir():
            raise ValueError("The input is a directory but not indicated with -d.")
        else:
            fasta_file = args.fasta
            extract_and_save(fasta_file=fasta_file, model=model, alphabet=alphabet, args=args)

# def main():
#     parser = create_parser()
#     args = parser.parse_args()
#     run(args)

# if __name__ == "__main__":
#     main()
