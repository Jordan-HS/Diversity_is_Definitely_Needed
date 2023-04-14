import subprocess
import argparse
import random
import os

parser = argparse.ArgumentParser()

# General dataset parameters
parser.add_argument("--classes", dest="classes", nargs='+', required=True)
parser.add_argumnet("--trick", dest="trick", type=str, required=True)
parser.add_argument("--outdir", dest="outdir", type=str, required=True)

# Multidomain parameters
parser.add_argument("--domains", dest="domains", nargs='+')

# Random scale parameters
parser.add_argument("--min_scale", dest="min_scale", type=int, default=1)
parser.add_argument("--max_scale", dest="max_scale", type=int, default=5)

# Stable Diffusion parameters
parser.add_argument("--n_samples", dest="n_samples", type=int, default=2)
parser.add_argument("--n_iter", dest="n_iter", type=int, default=1000)
parser.add_argument("--ddim_steps", dest="ddim_steps", type=int, default=40)
parser.add_argument("--seed", dest="seed", type=int, default=64)
parser.add_argument("--H", dest="H", type=int, default=512)
parser.add_argument("--W", dest="W", type=int, default=512)

args = parser.parse_args()

def remove_spaces(folder_name):
    if "_" in folder_name:
        folder_name = folder_name.replace("_", " ")
    return folder_name

assert args.trick in ["class_prompt", "multidomain", "random_scale"]

if args.trick == "class_prompt":
    for c in args.classes:
        c = remove_spaces(c)      
        
        subprocess.run(["python", "scripts/txt2img.py", "--prompt", f"{c}", "--H", args.H, "--W", args.W,
                    "--seed", args.seed, "--n_iter", args.n_iter, "--n_samples", args.n_samples, "--ddim_steps", 
                    args.ddim_steps, "--skip_grid", "--outdir", os.path.join(args.outdir, f"{c}")])

elif args.trick == "multidomain":

    domains = args.domains

    for c in args.classes:
        folder_name = c
        
        domain = random.choice(domains)
        c = remove_spaces(c)
            
        for domain in domains:
            
            # Edit this prompt if you wish to use a different format of multidomain generation
            # E.g. "a satellite “a satellite photo of a {class} in the style of a {domain}” for 
            # satellite generation 
            multi_domain_prompt = f"a {domain} of a {c}"

            subprocess.run(["python", "scripts/txt2img.py", "--prompt", multi_domain_prompt, "--H", args.H, "--W", args.W,
                    "--seed", args.seed, "--n_iter", args.n_iter, "--n_samples", args.n_samples, "--ddim_steps", 
                    args.ddim_steps, "--skip_grid", "--outdir", os.path.join(args.outdir, f"{c}")])

elif args.trick == "random_scale":
    random_scale = random.randint(args.min_scale, args.max_scale)

    for c in args.classes:
        c = remove_spaces(c)
        
        subprocess.run(["python", "scripts/txt2img.py", "--prompt", f"an image of {c}", "--H", args.H, "--W", args.W,
                    "--seed", args.seed, "--n_iter", args.n_iter, "--n_samples", args.n_samples, "--ddim_steps", 
                    args.ddim_steps, "--skip_grid", "--outdir", os.path.join(args.outdir, f"{c}"), "--scale", random_scale])