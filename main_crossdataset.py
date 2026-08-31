"""NACL-MERC cross-dataset transfer learning.

Stage 1: self-supervised pretraining (MMAE + NACL) on the source dataset (default: MELD).
Stage 2: supervised fine-tuning and evaluation on the target dataset (default: IEMOCAP 4-way).

Examples:
    # Full pipeline: MELD pretraining -> IEMOCAP 4-way transfer
    python main_crossdataset.py --pretrain_dataset meld --finetune_dataset iemocap_4

    # MELD pretraining -> IEMOCAP 6-way transfer
    python main_crossdataset.py --pretrain_dataset meld --finetune_dataset iemocap

    # Evaluate the saved fine-tuned checkpoint on the test set only
    python main_crossdataset.py --finetune_dataset iemocap_4 --eval_only

Stage behavior is controlled by the config files (config/<dataset>_pretrain.yaml for
stage 1, config/<dataset>.yaml for stage 2): `from_begin` runs pretraining,
`do_finetune` runs fine-tuning, `from_scratch` skips pretraining and trains the
encoder from random initialization, and `unimodal_inference` evaluates the saved
fine-tuned model under missing-modality conditions (set `modalities` accordingly).
"""

import argparse
import os
from datetime import datetime as dt

import torch

import graphdata as gdt
import models
import utils
from data import get_MELD_loaders

utils.make_route('./log', 'train.log')
log = utils.get_logger('./log/train.log')


def load_pretraining_data(args):
    """Build (or load the cached) graph dataset used for self-supervised pretraining."""
    data_dir = os.path.join(os.getcwd(), args.data_dir_path, args.dataset)
    args.data = os.path.join(data_dir, "data_" + args.dataset + ".pkl")

    if args.dataset in ("iemocap", "iemocap_4"):
        graph_file = os.path.join(data_dir, "graph_trainset.pkl")
        if not os.path.exists(graph_file):
            dataset = utils.load_pkl(args.data)
            pretrainset = gdt.iemocap_4_graphDataset(dataset["train"], 'train', args)
            utils.save_pkl(pretrainset, graph_file)
        else:
            pretrainset = utils.load_pkl(graph_file)

    elif args.dataset == "meld":
        graph_file = os.path.join(data_dir, "graph_pretrainset.pkl")
        if not os.path.exists(graph_file):
            _, _, _, all_loader = get_MELD_loaders(args.batch_size, args.data)
            pretrainset = gdt.meld_graphDataset(all_loader, 'pretrain', args)
            utils.save_pkl(pretrainset, graph_file)
        else:
            pretrainset = utils.load_pkl(graph_file)

    else:
        raise ValueError(f"Unknown pretraining dataset name: {args.dataset}")

    args.n_max_utterances = pretrainset.n_max_utterances
    args.n_max_speakers = pretrainset.n_max_speakers
    return pretrainset


def load_finetuning_data(args):
    """Build (or load the cached) train/dev/test graph datasets for fine-tuning."""
    data_dir = os.path.join(os.getcwd(), args.data_dir_path, args.dataset)
    args.data = os.path.join(data_dir, "data_" + args.dataset + ".pkl")

    if args.dataset not in ("iemocap", "iemocap_4"):
        raise ValueError(f"Unsupported fine-tuning dataset: {args.dataset}")

    data = None
    sets = {}
    for split in ("train", "dev", "test"):
        graph_file = os.path.join(data_dir, f"graph_{split}set.pkl")
        if not os.path.exists(graph_file):
            if data is None:
                data = utils.load_pkl(args.data)
            sets[split] = gdt.iemocap_4_graphDataset(data[split], split, args)
            utils.save_pkl(sets[split], graph_file)
        else:
            sets[split] = utils.load_pkl(graph_file)

    args.n_max_utterances = sets["train"].n_max_utterances
    args.n_max_speakers = sets["train"].n_max_speakers
    return sets["train"], sets["dev"], sets["test"]


def main(args1, args2):
    utils.set_seed(args1.seed)

    load_finetuned_for_inference = (
        not args2.do_finetune and not args2.from_scratch and args2.unimodal_inference
    )

    pretrainset = None
    if args2.eval_only or load_finetuned_for_inference:
        # Inference-only: no pretraining stage, no pretraining data needed.
        args1.from_begin = False
    else:
        log.info("Load pretraining dataset... Name: " + args1.dataset)
        pretrainset = load_pretraining_data(args1)

    log.info("Load finetuning dataset... Name: " + args2.dataset)
    trainset, devset, testset = load_finetuning_data(args2)

    # Apply the missing-modality override only AFTER the graph datasets are built:
    # graphs must always be constructed with the full config modality set, and the
    # override only affects the model's inference-time modality masking.
    override = getattr(args2, "inference_modalities", None)
    if override:
        args2.modalities = override
        args2.num_degree = len(args2.relation_type)
        if args2.relation_type == "eam" and args2.modalities == "atv":
            args2.num_degree += 1

    log.debug("Building model...")
    finetuned_ckpt = f"./{args1.save_model_checkpoint}_{args2.dataset}/finetune_{args1.modalities}_best_model.pt"

    if args2.eval_only:
        # Coach loads the fine-tuned checkpoint and evaluates the test set.
        model = None
    elif load_finetuned_for_inference:
        # Missing-modality inference: reuse the fine-tuned model with a reduced modality set.
        model = torch.load(finetuned_ckpt, map_location=args2.device, weights_only=False)
        for name, param in model.named_parameters():
            if not name.startswith("encoder.") and not name.startswith("linear_fusion") and not name.startswith("classifier"):
                param.requires_grad = False
        model.args = args2

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters:     {total_params:,}")
    else:
        if args1.dataset == "iemocap":
            n_nodes = trainset.n_max_utterances
        else:
            n_nodes = max(trainset.n_max_utterances, testset.n_max_utterances)
        encoder = models.EmotionHeartEncoder(args1, n_nodes)
        decoder = models.EmotionHeartDecoder(args1)
        model = models.EmotionHeartModel(args1, encoder, decoder).to(args1.device)

    opt1, sched1 = None, None
    if pretrainset is not None and model is not None:
        opt1 = models.Optim(float(args1.learning_rate), int(args1.T), float(args1.max_grad_value),
                            float(args1.weight_decay), int(args1.epochs), len(pretrainset))
        opt1.set_parameters(model.parameters(), args1.optimizer)
        sched1 = opt1.get_scheduler(args1.scheduler)

    print(f"args_pretrain:\n{args1}\n")
    print(f"args_finetune:\n{args2}")
    coach = models.Coach(pretrainset, trainset, devset, testset, model, opt1, sched1, args1, args2, log)

    log.info("Start training...")
    ret = coach.train()

    if args2.do_finetune:
        metrics = {
            "best_dev_f1": ret[0],
            "best_dev_acc": ret[1],
            "best_epoch": ret[2],
            "train_losses": ret[4],
            "dev_losses": ret[5],
            "dev_f1s": ret[6],
            "test_f1s": ret[7],
            "dev_accs": ret[8],
            "test_accs": ret[9],
            "test_losses": ret[10],
            "best_test_f1": ret[11],
        }
        analysis_dir = os.path.join(os.getcwd(), args1.save_analysis_path + '_' + args2.dataset)
        timestamp = dt.now().strftime('%Y-%m-%d-%H-%M-%S')

        utils.plot_and_save_loss(ret[4], ret[5], ret[10],
                                 filename=os.path.join(analysis_dir, f"loss_plot_{timestamp}.png"))

        save_metrics_path = os.path.join(analysis_dir, f"metrics_{timestamp}.log")
        os.makedirs(os.path.dirname(save_metrics_path), exist_ok=True)
        with open(save_metrics_path, "w", encoding="utf-8") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NACL-MERC cross-dataset transfer learning")
    parser.add_argument(
        "--pretrain_dataset", type=str, default="meld",
        choices=["meld", "iemocap"],
        help="Source dataset for self-supervised pretraining (config/<name>_pretrain.yaml)."
    )
    parser.add_argument(
        "--finetune_dataset", type=str, default="iemocap_4",
        choices=["iemocap", "iemocap_4"],
        help="Target dataset for fine-tuning/evaluation (config/<name>.yaml)."
    )
    parser.add_argument("--config_dir", type=str, default="config", help="Directory containing YAML configs.")
    parser.add_argument("--pretrain_config", type=str, default=None,
                        help="Explicit stage-1 YAML path (default: <config_dir>/<pretrain_dataset>_pretrain.yaml). "
                             "e.g. config/meld_pretrain_6way.yaml for the IEMOCAP 6-way transfer.")
    parser.add_argument("--finetune_config", type=str, default=None,
                        help="Explicit stage-2 YAML path (default: <config_dir>/<finetune_dataset>.yaml).")
    parser.add_argument(
        "--relation_type", type=str, default="eam", choices=["e", "ea", "eam"],
        help="Graph relations. e: interlocutor, a: intralocutor, m: intermodality."
    )
    parser.add_argument("--pretrain_optimizer", type=str, default="adamw",
                        choices=["sgd", "rmsprop", "adam", "adamw"])
    parser.add_argument("--finetune_optimizer", type=str, default="adam",
                        choices=["sgd", "rmsprop", "adam", "adamw"])
    parser.add_argument("--device", type=str, default=None,
                        help="Device override for both stages (e.g. cuda:1).")
    parser.add_argument("--modalities", type=str, default=None,
                        choices=["a", "t", "v", "at", "tv", "av", "atv"],
                        help="Fine-tuning/inference modality override (used for missing-modality inference).")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and evaluate the saved fine-tuned checkpoint on the test set.")
    cli = parser.parse_args()

    pretrain_cfg = cli.pretrain_config or os.path.join(cli.config_dir, cli.pretrain_dataset + "_pretrain.yaml")
    finetune_cfg = cli.finetune_config or os.path.join(cli.config_dir, cli.finetune_dataset + ".yaml")
    args1 = utils.load_config(pretrain_cfg, cli.pretrain_dataset, cli.relation_type, cli.pretrain_optimizer, cli.device)
    args2 = utils.load_config(finetune_cfg, cli.finetune_dataset, cli.relation_type, cli.finetune_optimizer, cli.device)

    if cli.modalities is not None:
        if args2.do_finetune or not args2.unimodal_inference:
            parser.error(
                "--modalities is only for missing-modality inference: set "
                "do_finetune: false and unimodal_inference: true in the target config."
            )
        # Applied in main() after the graph datasets are built.
        args2.inference_modalities = cli.modalities

    args2.eval_only = cli.eval_only
    if cli.eval_only:
        args1.from_begin = False
        args2.do_finetune = False
        args2.from_scratch = False
        args2.unimodal_inference = False

    log.debug(args1)
    log.debug(args2)

    main(args1, args2)
