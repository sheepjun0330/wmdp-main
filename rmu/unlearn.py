import os
import datetime
from collections import defaultdict

import numpy as np
import torch
from torch.optim import AdamW
import tqdm as tqdm
import wandb

from rmu.utils import load_model, get_params, forward_with_cache, get_data


def _global_grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total += g.norm(2).item() ** 2
    return total ** 0.5

def _clip_grad(params, max_norm: float):
    if max_norm is None:
        return
    max_norm = float(max_norm)
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)


@torch.no_grad()
def _adamw_state_step_and_delta(
    p: torch.Tensor,
    g: torch.Tensor,
    state: dict,
    lr: float,
    betas: tuple,
    eps: float,
):
    beta1, beta2 = float(betas[0]), float(betas[1])

    if "step" not in state:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32, device=p.device)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32, device=p.device)

    state["step"] += 1
    t = int(state["step"])

    m = state["exp_avg"]
    v = state["exp_avg_sq"]

    gf = g.detach().float()

    m.mul_(beta1).add_(gf, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(gf, gf, value=1.0 - beta2)

    bc1 = 1.0 - (beta1 ** t)
    bc2 = 1.0 - (beta2 ** t)

    m_hat = m / max(bc1, 1e-30)
    v_hat = v / max(bc2, 1e-30)

    delta = (-lr) * (m_hat / (v_hat.sqrt().add_(eps)))
    return delta.to(dtype=p.dtype)

def compute_rmu_forget_loss(
    updated_model,
    updated_module,
    tokenizer,
    unlearn_batch,
    control_vec,
    max_length,
):
    unlearn_inputs = tokenizer(
        unlearn_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(updated_model.device)

    updated_forget_activations = forward_with_cache(
        updated_model, unlearn_inputs, module=updated_module, no_grad=False
    ).to(updated_model.device)

    unlearn_loss = torch.nn.functional.mse_loss(
        updated_forget_activations, control_vec
    )
    return unlearn_loss, unlearn_inputs, updated_forget_activations

def compute_rmu_retain_loss(
    updated_model,
    frozen_model,
    updated_module,
    frozen_module,
    tokenizer,
    retain_batch,
    alpha,
    max_length=512,
):
    retain_inputs = tokenizer(
        retain_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(updated_model.device)

    updated_retain_activations = forward_with_cache(
        updated_model, retain_inputs, module=updated_module, no_grad=False
    ).to(updated_model.device)

    frozen_retain_activations = forward_with_cache(
        frozen_model, retain_inputs, module=frozen_module, no_grad=True
    ).to(updated_model.device)

    retain_loss = torch.nn.functional.mse_loss(
        updated_retain_activations, frozen_retain_activations
    )
    retain_loss = retain_loss * alpha

    return retain_loss, retain_inputs, updated_retain_activations, frozen_retain_activations



def sam_adv_grad_no_step(
    params,
    loss_closure,
    rho,
    clip_norm=0.0,
    perturb_sign=+1.0,
):
    for p in params:
        p.grad = None

    loss_clean = loss_closure()
    loss_clean.backward()
    _clip_grad(params, clip_norm)
    grad_norm_clean = _global_grad_norm(params)

    eps_list = []
    with torch.no_grad():
        denom = 0.0
        for p in params:
            if p.grad is not None:
                denom += (p.grad.detach().float().norm() ** 2)
        denom = denom.sqrt().clamp_min(1e-12)

        for p in params:
            if p.grad is None:
                eps_list.append(None)
                continue
            eps_ = (perturb_sign * float(rho)) * (p.grad.detach() / denom)
            p.add_(eps_)
            eps_list.append(eps_)

    for p in params:
        p.grad = None

    loss_adv = loss_closure()
    loss_adv.backward()
    _clip_grad(params, clip_norm)

    g_list = []
    for p in params:
        g_list.append(None if p.grad is None else p.grad.detach().clone())

    with torch.no_grad():
        for p, eps_ in zip(params, eps_list):
            if eps_ is not None:
                p.sub_(eps_)

    for p in params:
        p.grad = None

    return g_list, loss_clean.detach(), loss_adv.detach(), float(grad_norm_clean)

def run_rmu(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    rmu_config = vars(args)
    print("====rmu Config====")
    print("\n".join(f"{k}={v}" for k,v in rmu_config.items()))
    print("=====")

    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)


    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )


    adam_state_r = defaultdict(dict)
    adam_state_f = defaultdict(dict)

    lam = torch.tensor(
        args.lagran_lambda_init,
        device=updated_model.device,
        dtype=torch.float64,
    ).clamp(min=0.0)
    
    frozen_module = eval(
        args.module_str.format(model_name="frozen_model", layer_id=args.layer_id)
    )
    updated_module = eval(
        args.module_str.format(model_name="updated_model", layer_id=args.layer_id)
    )

    control_vectors_list = []
    for i in range(len(forget_data_list)):
        random_vector = torch.rand(1,1, updated_model.config.hidden_size, dtype=updated_model.dtype, device=updated_model.device)
        control_vec = random_vector / torch.norm(random_vector) * args.steering_coeff_list[i]
        control_vectors_list.append(control_vec)

    num_batches = min(
        args.max_num_batches,
        min([len(f) for f in forget_data_list]),
        min([len(r) for r in retain_data_list]),
    )
    
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side="right"

    for epoch in range(1):
        print(f"======= Epoch {epoch} =======")
        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                topic_idx = idx % len(forget_data_list)
                batch_idx = idx // len(forget_data_list)
                control_vec = control_vectors_list[topic_idx]
                unlearn_batch = forget_data_list[topic_idx][batch_idx]
                retain_batch = retain_data_list[topic_idx][batch_idx]

                # Unlearning loss
                max_length = 512 if topic_idx == 0 else 768

                def forget_loss_closure():
                    loss, _, _ = compute_rmu_forget_loss(
                        updated_model=updated_model,
                        updated_module=updated_module,
                        tokenizer=tokenizer,
                        unlearn_batch=unlearn_batch,
                        control_vec=control_vec,
                        max_length=max_length,
                    )
                    return loss

                def retain_loss_closure():
                    loss, _, _, _ = compute_rmu_retain_loss(
                        updated_model=updated_model,
                        frozen_model=frozen_model,
                        updated_module=updated_module,
                        frozen_module=frozen_module,
                        tokenizer=tokenizer,
                        retain_batch=retain_batch,
                        alpha=args.alpha[topic_idx],
                        max_length=512,
                    )
                    return loss
                
                if args.dual_mode != "alm_sam_sam_joint2":
                    raise ValueError(
                        f"Currently only dual_mode=alm_sam_sam_joint2 is implemented, got {args.dual_mode}"
                    )

                # retain SAM adversarial gradient
                g_r, retain_loss_clean, retain_loss_adv, gn_r_clean = sam_adv_grad_no_step(
                    params=params,
                    loss_closure=retain_loss_closure,
                    rho=args.retain_rho,
                    clip_norm=args.retain_clip_norm,
                    perturb_sign=+1.0,
                )

                # forget SAM adversarial gradient
                g_f, unlearn_loss_clean, unlearn_loss_adv, gn_f_clean = sam_adv_grad_no_step(
                    params=params,
                    loss_closure=forget_loss_closure,
                    rho=args.forget_rho,
                    clip_norm=args.forget_clip_norm,
                    perturb_sign=+1.0,
                )

                # RMU에서는 forget loss도 minimize 대상
                # constraint: Lf_adv <= tau
                constraint = float(unlearn_loss_adv.item() - args.tau)
                lam_next = max(
                    float(lam.item()) + args.lagran_lambda_lr * constraint,
                    0.0,
                )
                coef = 1.0 + float(lam_next)

                with torch.no_grad():
                    if args.weight_decay > 0.0:
                        shrink = 1.0 - args.lr * args.weight_decay
                        for p in params:
                            p.mul_(shrink)

                    for i, p in enumerate(params):
                        gr = g_r[i]
                        gf = g_f[i]

                        if gr is not None:
                            d_r = _adamw_state_step_and_delta(
                                p=p,
                                g=gr,
                                state=adam_state_r[p],
                                lr=args.lr,
                                betas=args.retain_betas,
                                eps=args.adam_epsilon,
                            )
                            p.add_(d_r*2)

                        if coef > 0.0 and gf is not None:
                            d_f = _adamw_state_step_and_delta(
                                p=p,
                                g=gf,
                                state=adam_state_f[p],
                                lr=args.lr,
                                betas=args.forget_betas,
                                eps=args.adam_epsilon,
                            )
                            p.add_(d_f, alpha=coef)

                    lam.fill_(lam_next)

                with torch.no_grad():
                    unlearn_loss, unlearn_inputs, updated_forget_activations = compute_rmu_forget_loss(
                        updated_model=updated_model,
                        updated_module=updated_module,
                        tokenizer=tokenizer,
                        unlearn_batch=unlearn_batch,
                        control_vec=control_vec,
                        max_length=max_length,
                    )

                    retain_loss, retain_inputs, updated_retain_activations, frozen_retain_activations = compute_rmu_retain_loss(
                        updated_model=updated_model,
                        frozen_model=frozen_model,
                        updated_module=updated_module,
                        frozen_module=frozen_module,
                        tokenizer=tokenizer,
                        retain_batch=retain_batch,
                        alpha=args.alpha[topic_idx],
                        max_length=512,
                    )

                print(
                    f"loss={(retain_loss + coef * unlearn_loss).item():.4g} | "
                    f"unlearn_clean={unlearn_loss.item():.4g} | "
                    f"unlearn_adv={unlearn_loss_adv.item():.4g} | "
                    f"retain_clean={retain_loss.item():.4g} | "
                    f"retain_adv={retain_loss_adv.item():.4g} | "
                    f"lambda={lam.item():.4g} | "
                    f"constraint=(Lf_adv-tau)={constraint:.4g} | "
                    f"gn_f={gn_f_clean:.4g} | "
                    f"gn_r={gn_r_clean:.4g}"
                )


                # print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")

                if args.use_wandb:
                    wandb.log({
                        "loss": (retain_loss + coef * unlearn_loss).item(),
                        "unlearn_clean": unlearn_loss.item(),
                        "unlearn_adv": unlearn_loss_adv.item(),
                        "retain_clean": retain_loss.item(),
                        "retain_adv": retain_loss_adv.item(),
                        "lambda": lam.item(),
                        "constraint": constraint,
                        "grad_norm_forget": gn_f_clean,
                        "grad_norm_retain": gn_r_clean,
                    })
                
                # ======= Logging ======
                if args.verbose:
                    frozen_forget_activations = forward_with_cache(frozen_model, unlearn_inputs, module=frozen_module, no_grad=True).to(updated_model.device)
                    unlearn_cosine= torch.nn.functional.cosine_similarity(updated_forget_activations, frozen_forget_activations, dim=-1).mean()
                    retain_cosine = torch.nn.functional.cosine_similarity(updated_retain_activations, frozen_retain_activations, dim=-1).mean()
                    
                    print(f"unlearn_cosine_sim={unlearn_cosine.item()}")
                    print(f"retain_cosine_sim={retain_cosine.item()}")
                    print(f"Topic {topic_idx} updated_forget_activations.norm=",torch.mean(updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"Topic {topic_idx} frozen_forget_activations.norm=",torch.mean(frozen_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"Topic {topic_idx} updated_retain_activations.norm=",torch.mean(updated_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"Topic {topic_idx} frozen_retain_activations.norm=",torch.mean(frozen_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item())

                pbar.update(1)

    tokenizer.truncation_side = truncation_side
    # Save model
    if args.output_dir:
        path = args.output_dir
    else:
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = f"models/{args.model_name_or_path}_alpha-{args.alpha}_batches-{num_batches}_layer-{args.layer_id}_{date}"
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")
    if args.use_wandb:
        wandb.finish()  


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    ### Model arguments
    parser.add_argument(
        "--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta"
    )
    parser.add_argument(
        "--module_str", type=str, default="{model_name}.model.layers[{layer_id}]"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None
    )
    ### Data arguments
    parser.add_argument(
        "--retain_corpora",
        type=str,
        default="wikitext,wikitext",
        help="comma-separated list of corpora to retain",
    )
    parser.add_argument(
        "--forget_corpora",
        type=str,
        default="bio-forget-corpus,cyber-forget-corpus",
        help="comma-separated list of corpora to forget",
    )
    ### rmu hyperparameters
    parser.add_argument("--alpha", type=str, default="100,100", help="retain weight")
    parser.add_argument(
        "--steering_coeffs",
        type=str,
        default="20,20",
        help="Steer vector weight in order of topic",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=80)
    parser.add_argument("--layer_id", type=int, default=7, help="layer to unlearn")
    parser.add_argument("--layer_ids", type=str, default="5,6,7", help="update layers")
    parser.add_argument("--param_ids", type=str, default="6", help="update params")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--verbose", action="store_true", help="Logging the activations norms and cosine at each step")

    # ===== dual / ALM / SAM args =====
    parser.add_argument("--dual_mode", type=str, default="alm_sam_sam_joint2")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lagran_lambda_init", type=float, default=0.0)
    parser.add_argument("--lagran_lambda_lr", type=float, default=1e-3)

    parser.add_argument("--forget_rho", type=float, default=5e-3)
    parser.add_argument("--retain_rho", type=float, default=5e-3)

    parser.add_argument("--forget_betas", type=str, default="0.9,0.999")
    parser.add_argument("--retain_betas", type=str, default="0.9,0.999")

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    parser.add_argument("--forget_clip_norm", type=float, default=0.0)
    parser.add_argument("--retain_clip_norm", type=float, default=0.0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rmu-unlearn")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    args.retain_corpora = args.retain_corpora.split(",")
    args.forget_corpora = args.forget_corpora.split(",")
    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
    args.alpha = [float(c) for c in args.alpha.split(",")]
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]

    args.forget_betas = tuple(float(x) for x in args.forget_betas.split(","))
    args.retain_betas = tuple(float(x) for x in args.retain_betas.split(","))

    return args 


if __name__ == "__main__":
    args = get_args()

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, tokenizer = load_model(args.model_name_or_path)
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.max_len,
        args.batch_size,
    )
    run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )