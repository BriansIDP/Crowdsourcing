import json
import numpy as np
from scipy.stats import spearmanr, pearsonr

from EM import EM_orig, EM_bimodal, EM_bimodal_biased

def get_crosscheck_data(llm_to_rank, evidence_llm):

    if llm_to_rank not in evidence_llm:
        evidence_llm = evidence_llm + [llm_to_rank]

    # Get selfcheck scores
    selfcheckscores = {}
    cov_crosscheck = {}
    for llm in evidence_llm:
        selfcheckscores[llm] = []
        cov_crosscheck[llm] = {}
        with open("data/crosscheck/crosscheck_prompt_{}.json".format(llm)) as fin:
            data = json.load(fin)
        if llm == "gpt3":
            data = data["results"]
        for paragraph in data:
            selfcheckscores[llm].append(sum(paragraph[llm]) / len(paragraph[llm]) / 20)
            for ellm in evidence_llm:
                if ellm not in cov_crosscheck[llm]:
                    cov_crosscheck[llm][ellm] = []
                cov_crosscheck[llm][ellm].append(sum(paragraph[ellm]) / len(paragraph[ellm]) / 20)
        # selfcheckscores[llm] = sum(selfcheckscores[llm]) / len(selfcheckscores[llm]) / 20
    selfcheckscores = np.array([selfcheckscores[llm] for llm in evidence_llm]).transpose()
    cov_crosscheck = [[cov_crosscheck[llm][ellm] for ellm in evidence_llm] for llm in evidence_llm]
    cov_crosscheck = np.array(cov_crosscheck).transpose(2, 0, 1)

    with open("data/crosscheck/crosscheck_prompt_{}.json".format(llm_to_rank)) as fin:
        data = json.load(fin)
    # Get refcheck scores
    refcheckscores = []
    if llm_to_rank == "gpt3":
        mapping = {
            "major_inaccurate": 1,
            "minor_inaccurate": 0.5,
            "accurate": 0,
        }
        for paragraph in data["labels"]:
            para_score = [mapping[score] for score in paragraph]
            refcheckscores.append(sum(para_score)/len(para_score))
    else:
        with open("data/crosscheck/crosscheck_prompt_{}_refcheck.json".format(llm_to_rank)) as fin:
            refdata = json.load(fin)
        for paragraph in refdata:
            refcheckscores.append(sum(paragraph["ref"]) / len(paragraph["ref"]))
    refcheckscores = np.array(refcheckscores)

    # Get crosscheck scores
    crosscheckscores = {}
    if llm_to_rank == "gpt3":
        data = data["results"]
    for i, paragraph in enumerate(data):
        for llm in evidence_llm:
            if llm not in crosscheckscores:
                crosscheckscores[llm] = []
            crosscheckscores[llm].append(sum(paragraph[llm]) / len(paragraph[llm]) / 20)
    crosscheckscores = np.array([crosscheckscores[llm] for llm in evidence_llm]).transpose()
    return cov_crosscheck, crosscheckscores, refcheckscores


def test_crosscheck(crosscheckscore, refcheckscore, cov_crosscheck, caliberation=10000):
    n_systems = crosscheckscore.shape[1]
    print("="*89)
    print("SelfCheckGPT baseline: {:.5f}".format(pearsonr(refcheckscore, crosscheckscore[:, -1])[0]))
    import pdb; pdb.set_trace()
    simple_ave = crosscheckscore.mean(axis=-1)
    print("Unweighted average Passage level: {:.5f}".format(pearsonr(refcheckscore, simple_ave)[0]))
    selfcheck = cov_crosscheck.diagonal(0, 1, 2).mean(axis=0) / caliberation
    inverse_var_ave_weight = np.exp(selfcheck) / np.exp(selfcheck).sum(axis=-1)
    # inverse_var_ave_weight = np.array([0.1, 0.1, 0, 1])
    inverse_var_ave = np.matmul(crosscheckscore, inverse_var_ave_weight[:, None])
    print("Inverse variance average Passage level: {:.5f}".format(pearsonr(refcheckscore, inverse_var_ave[:, 0])[0]))
    print("="*89)
    logdata = - np.log(1 / (crosscheckscore+1e-5) - 1 + 1e-5)
    # EM_ave, weight, Sigma_hat = EM_orig(logdata, n_systems, sigma_bar=2, rho_bar=0, c=0.1, M=10000, v_bar=10, mu_bar=0)
    EM_ave, weight, Sigma_hat = EM_bimodal(logdata, n_systems, sigma_bar=2, rho_bar=0, c=0.1, M=10000, v_bar=2, mu_bar=0)
    print("Inverse covariance weights:", weight)
    EM_ave = 1 / (np.exp(- EM_ave) + 1 - 1e-5) - 1e-5
    print("Inverse covariance average Passage level: {:.5f}".format(pearsonr(refcheckscore, EM_ave)[0]))
    print("="*89)
    # EM_ave, Sigma_hat_pos, Sigma_hat_neg, m_hat_pos, m_hat_neg = EM_bimodal_biased(
    #     logdata,
    #     sigma_bar=2,
    #     rho_bar=0.0,
    #     c=0,
    #     M=10000,
    #     v_bar=5,
    #     mu_bar=0,
    #     assign="mean",
    #     m_bar=0,
    #     tied=True,
    # )
    # EM_ave = 1 / (np.exp(- EM_ave) + 1 - 1e-5) - 1e-5
    # print("Predicted pos means: {}".format(m_hat_pos))
    # print("Predicted neg means: {}".format(m_hat_neg))
    # print("Biased Gaussian noise average Passage level: {:.5f}".format(pearsonr(refcheckscore, EM_ave)[0]))
    # print("="*89)
    return


if __name__ == "__main__":
    llm_to_rank = "vicuna"

    evidence_llm = ["mistral", "llama2", "vicuna", "zephyr", "beluga", "starling", "openorca", "llama2lm"]
    # evidence_llm = ["mistral", "llama2", "vicuna", "beluga", "starling", "openorca", "gpt3"]
    print("LLM under checking: {}".format(llm_to_rank))
    print(evidence_llm)

    caliberation = 0.1
    cov_crosscheck, crosscheckscore, refcheckscore = get_crosscheck_data(llm_to_rank, evidence_llm)
    np.save("outputs/crosscheck.npy", crosscheckscore)
    np.save("outputs/refcheck.npy", refcheckscore)
    test_crosscheck(crosscheckscore, refcheckscore, cov_crosscheck, caliberation=caliberation)
