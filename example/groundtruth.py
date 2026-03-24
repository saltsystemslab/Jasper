import jasper

print("reading vectors")
vectors = jasper.read_bin("/projects/SaltSystemsLab/ann_data/deep/deep200M", dtype="f32").cpu()
queries = jasper.read_bin("/projects/SaltSystemsLab/ann_data/deep/deep10kquery", dtype="f32").cpu()
print("finished reading vectors")

gt_ids, gt_dists = jasper.generate_groundtruth(
    vectors, queries, k=100, distance="l2",
    query_batch_size=128, vector_batch_size=200_000,
)
jasper.save_groundtruth("/projects/SaltSystemsLab/ann_gt/deep200M-gt", gt_ids, gt_dists)