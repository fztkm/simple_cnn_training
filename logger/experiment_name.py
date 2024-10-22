from datetime import datetime


def create_exp_name(args):
    exp_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    exp_name = exp_name.replace(" ", "_")

    tags = []
    if args.val_only:
        tags.append("val")
    tags.append(f"lr{args.lr}")

    # tags.append(f"ps{args.patch_size}")

    tags.append(f"bs({args.batch_size}x{args.devices}-ac{args.grad_accum})")
    exp_name += "_" + "_".join(tags)
    return exp_name, tags
