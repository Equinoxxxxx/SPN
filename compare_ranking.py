from scipy.stats import rankdata


def values_to_ranks(values, descending=True):
    data = [-v for v in values] if descending else values
    return rankdata(data, method='average')


def sorted_indices_to_ranks(sorted_indices):
    n = len(sorted_indices)
    ranks = [0] * n
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank + 1  # 排名从1开始
    return ranks


def spearman_rank_correlation(rank_a, rank_b):
    if len(rank_a) != len(rank_b):
        raise ValueError("rank_a and rank_b must have the same length")
    n = len(rank_a)
    d_squared_sum = sum((a - b) ** 2 for a, b in zip(rank_a, rank_b))
    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return rho


if __name__ == '__main__':
    from llm_eval.prompts import proto_rank_topkms, llm_eval_res
    import numpy as np
    llm_score_726_cld = llm_eval_res['726']['claude']['claude_llm_eval.log']
    llm_score_cld_726_08Apr2025_16h48m52s = llm_eval_res['726']['claude']['claude_llm_eval_08Apr2025-16h48m52s_.log']
    llm_score_726_gpt4o = llm_eval_res['726']['gpt-4o']['gpt-4o_llm_eval_07Apr2025-23h52m46s.log']
    llm_score_726_gpt45 = llm_eval_res['726']['gpt-4.5-preview']['gpt-4.5-preview_llm_eval_08Apr2025-00h21m17s.log']
    llm_score_717_cld = llm_eval_res['717']['claude']['claude_llm_eval_10Apr2025-18h44m34s.log']
    llm_score_717_gpt4o = llm_eval_res['717']['gpt-4o']['gpt-4o_llm_eval_10Apr2025-19h55m40s.log']
    llm_score_717_cld_35 = llm_eval_res['717']['claude']['claude-3-5-sonnet-latest_llm_eval_10Apr2025-20h12m44s.log']
    print(f'mean and std of llm_score_cld: {np.mean(llm_score_726_cld)}, {np.std(llm_score_726_cld)}')
    print(f'mean and std of llm_score_cld_08Apr2025_16h48m52s: {np.mean(llm_score_cld_726_08Apr2025_16h48m52s)}, {np.std(llm_score_cld_726_08Apr2025_16h48m52s)}')
    print(f'mean and std of llm_score_gpt4o: {np.mean(llm_score_726_gpt4o)}, {np.std(llm_score_726_gpt4o)}')
    print(f'mean and std of llm_score_gpt45: {np.mean(llm_score_726_gpt45)}, {np.std(llm_score_726_gpt45)}')
    print(f'mean and std of llm_score_717_cld: {np.mean(llm_score_717_cld)}, {np.std(llm_score_717_cld)}')
    print(f'mean and std of llm_score_717_gpt4o: {np.mean(llm_score_717_gpt4o)}, {np.std(llm_score_717_gpt4o)}')
    print(f'mean and std of llm_score_717_cld_35: {np.mean(llm_score_717_cld_35)}, {np.std(llm_score_717_cld_35)}')
    topkms_sorted_idx_726 = proto_rank_topkms['726']
    topkms_ranks_726 = sorted_indices_to_ranks(topkms_sorted_idx_726)
    topkms_sorted_idx_717 = proto_rank_topkms['717']
    topkms_ranks_717 = sorted_indices_to_ranks(topkms_sorted_idx_717)
    llm_ranks_726_cld = values_to_ranks(llm_score_726_cld)
    llm_ranks_726_gpt4o = values_to_ranks(llm_score_726_gpt4o)
    llm_ranks_726_gpt45 = values_to_ranks(llm_score_726_gpt45)
    llm_ranks_726_cld_08Apr2025_16h48m52s = values_to_ranks(llm_score_cld_726_08Apr2025_16h48m52s)
    llm_ranks_717_cld = values_to_ranks(llm_score_717_cld)
    llm_ranks_717_gpt4o = values_to_ranks(llm_score_717_gpt4o)
    llm_ranks_717_cld_35 = values_to_ranks(llm_score_717_cld_35)
    print(f'726 cld: {spearman_rank_correlation(topkms_ranks_726, llm_ranks_726_cld)}')
    print(f'726 cld 08Apr2025_16h48m52s: {spearman_rank_correlation(topkms_ranks_726, llm_ranks_726_cld_08Apr2025_16h48m52s)}')
    print(f'726 gpt4o: {spearman_rank_correlation(topkms_ranks_726, llm_ranks_726_gpt4o)}')
    print(f'726 gpt45: {spearman_rank_correlation(topkms_ranks_726, llm_ranks_726_gpt45)}')
    print(f'717 cld 37: {spearman_rank_correlation(topkms_ranks_717, llm_ranks_717_cld)}')
    print(f'717 gpt4o: {spearman_rank_correlation(topkms_ranks_717, llm_ranks_717_gpt4o)}')
    print(f'717 cld 35: {spearman_rank_correlation(topkms_ranks_717, llm_ranks_717_cld_35)}')