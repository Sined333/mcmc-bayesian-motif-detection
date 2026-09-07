import numpy as np
from pysam import FastaFile
import matplotlib.pyplot as plt



# Lecture des séquences ====================================================================
def load_sequences(filepath):
    with FastaFile(filepath) as fastafile:
        sequences = []
        names = fastafile.references
        for seqname in names:
            sequences.append(fastafile.fetch(seqname))
    return sequences, names


# Estimation de Phi via une CM par maximum de vraisemblance ===============================
# Degree 1
def estimate_Phi(sequences):
    P = np.zeros((4, 4))
    nb_occurence_P = np.zeros(4)
    u = np.zeros(4)
    nb_seq = 0
    for seq in sequences:
        nb_seq += 1
        u[seq[0]] += 1
        for i in range(len(seq) - 1):
            current_nuc = seq[i]
            next_nuc = seq[i+1]
            
            nb_occurence_P[current_nuc] += 1
            P[current_nuc, next_nuc] += 1

    u /= nb_seq
    for i in range(4):
        if nb_occurence_P[i] > 0:
            P[i] /= nb_occurence_P[i]

    return u, P

# Echantillonnage de Theta | A, S  via la distribution de Dirichlet
def sample_Theta(sequences, A, W, alpha):
    Theta = np.zeros((4, W))
    for k in range(len(sequences)):
        ak = A[k]
        pattern = sequences[k][ak: ak + W]
        Theta[pattern, np.arange(W)] += 1

    for j in range(W):
        alpha_post = Theta[:, j] + alpha
        Theta[:, j] = np.random.dirichlet(alpha_post)
    return Theta


# Echantillonnage de Ak | A_{-k}, Theta, S
def sample_Ak(seq, Theta, u, P, W):
    L = len(seq)
    n_pos = L - W + 1
    log_Theta = np.log(Theta)

    log_trans = np.zeros(L)
    log_trans[0] = np.log(u[seq[0]])
    log_trans[1:] = np.log(P[seq[:-1], seq[1:]])
    cumsum = np.cumsum(log_trans)

    indices = np.arange(n_pos)[:, None] + np.arange(W)[None, :]
    motif_scores = np.sum(log_Theta[seq[indices], np.arange(W)], axis=1)

    before_scores = np.zeros(n_pos)
    before_scores[1:] = cumsum[:n_pos - 1]

    after_scores = np.zeros(n_pos)
    for a in range(n_pos):
        if a + W < L:
            after_scores[a] = log_trans[a + W] + (cumsum[L-1] - cumsum[a + W])

    log_scores = before_scores + motif_scores + after_scores
    log_scores -= np.max(log_scores)
    probs = np.exp(log_scores)
    probs /= probs.sum()
    return np.random.choice(n_pos, p=probs)
"""
# degree 2
def sample_Ak(seq, Theta, u, P, W):
    L = len(seq)
    n_pos = L - W + 1
    log_Theta = np.log(Theta)

    u_1d = np.sum(u, axis=1)
    u_1d /= np.sum(u_1d)

    log_trans = np.zeros(L)
    log_trans[1] = np.log(u[seq[0], seq[1]])
    log_trans[2:] = np.log(P[seq[:-2], seq[1:-1], seq[2:]])
    cumsum = np.cumsum(log_trans)

    indices = np.arange(n_pos)[:, None] + np.arange(W)[None, :]
    motif_scores = np.sum(log_Theta[seq[indices], np.arange(W)], axis=1)

    before_scores = np.zeros(n_pos)
    before_scores[1] = np.log(u_1d[seq[0]])
    before_scores[2:] = cumsum[1:n_pos - 1]

    after_scores = np.zeros(n_pos)
    for a in range(n_pos):
        if a + W < L:
            after_scores[a] += np.log(P[seq[a + W - 2], seq[a + W - 1], seq[a + W]])
            if a + W + 1 < L:
                after_scores[a] += np.log(P[seq[a + W - 1], seq[a + W], seq[a + W + 1]])
                if a + W + 2 < L:
                    after_scores[a] += cumsum[L - 1] - cumsum[a + W + 1]

    log_scores = before_scores + motif_scores + after_scores
    log_scores -= np.max(log_scores)
    probs = np.exp(log_scores)
    probs /= probs.sum()
    return np.random.choice(n_pos, p=probs)
"""

# Log-vraisemblance générale: p(S | A, Theta, Phi)

def log_likelihood(sequences, positions, Theta, u, P, W):
    log_Theta = np.log(Theta)
    lp = 0.0
    
    for k, seq in enumerate(sequences):
        ak = positions[k]
        L = len(seq)
        
        log_trans = np.zeros(L)
        log_trans[0] = np.log(u[seq[0]])
        log_trans[1:] = np.log(P[seq[:-1], seq[1:]])
        cumsum = np.cumsum(log_trans)
        
        if ak > 0:
            lp += cumsum[ak - 1]
        
        motif_indices = np.arange(W)
        lp += np.sum(log_Theta[seq[ak:ak + W], motif_indices])
        
        if ak + W < L:
            lp += cumsum[L - 1] - cumsum[ak + W - 1]
    
    return lp
"""

#degree 2
def log_likelihood(sequences, positions, Theta, u, P, W):
    log_Theta = np.log(Theta)
    lp = 0.0
    
    u_1d = np.sum(u, axis=1)
    u_1d /= np.sum(u_1d) 

    for k, seq in enumerate(sequences):
        ak = positions[k]
        L = len(seq)
        
        log_trans = np.zeros(L)
        log_trans[1] = np.log(u[seq[0], seq[1]])
        log_trans[2:] = np.log(P[seq[:-2], seq[1:-1], seq[2:]])
        cumsum = np.cumsum(log_trans)
        
        if ak == 1:
            lp += np.log(u_1d[seq[0]])
        elif ak > 1:
            lp += cumsum[ak - 1]
        
        motif_indices = np.arange(W)
        lp += np.sum(log_Theta[seq[ak:ak + W], motif_indices])
        
        if ak + W < L:
            lp += np.log(P[seq[ak + W - 2], seq[ak + W - 1], seq[ak + W]]) 
            if ak + W + 1 < L:
                lp += np.log(P[seq[ak + W - 1], seq[ak + W], seq[ak + W + 1]])
                if ak + W + 2 < L:
                    lp += cumsum[L - 1] - cumsum[ak + W + 1]
    
    return lp
"""
# Algorithme de Gibbs classic
def gibbs(sequences, W, nb_iter, alpha=1.0):
    u, P = estimate_Phi(sequences)
    A = np.zeros(len(sequences), dtype=int)
    for i, seq in enumerate(sequences):
        A[i] = np.random.randint(0, len(seq) - W + 1)

    Theta = sample_Theta(sequences, A, W, alpha)

    best_A = A[:]
    best_Theta = Theta.copy()
    best_ll = log_likelihood(sequences, A, Theta, u, P, W)

    lls = []
    for n in range(nb_iter):
        random_k_list = np.random.permutation(len(sequences))
        for k in random_k_list:
            A[k] = sample_Ak(sequences[k], Theta, u, P, W)

        Theta = sample_Theta(sequences, A, W, alpha)
        ll = log_likelihood(sequences, A, Theta, u, P, W)
        if ll > best_ll:
            best_ll = ll
            best_A = A[:]
            best_Theta = Theta.copy()
        lls.append(ll)
    return best_A, best_Theta, lls


# Fonction de décalage pour gibbs_with_shift
def shift(sequences, A, Theta, u, P, W, delta, alpha=1.0):
    A_new = A + delta
    for k, seq in enumerate(sequences):
        if A_new[k] < 0 or A_new[k] > len(seq) - W:
            return A, Theta
    
    Theta_new = sample_Theta(sequences, A_new, W, alpha)
    ll_current = log_likelihood(sequences, A, Theta, u, P, W)
    ll_new = log_likelihood(sequences, A_new, Theta_new, u, P, W)
    log_ratio = ll_new - ll_current
    
    if np.log(np.random.uniform()) < log_ratio:
        return A_new, Theta_new
    else:
        return A, Theta


# Algorithme de Gibbs avec décalage
def gibbs_with_shift(sequences, W, nb_iter, alpha=1, shift_freq=10):
    u, P = estimate_Phi(sequences)
    A = np.zeros(len(sequences), dtype=int)
    for i, seq in enumerate(sequences):
        A[i] = np.random.randint(0, len(seq) - W + 1)

    Theta = sample_Theta(sequences, A, W, alpha)

    best_A = A[:]
    best_Theta = Theta.copy()
    best_ll = log_likelihood(sequences, A, Theta, u, P, W)
    lls = []
    for n in range(nb_iter):
        random_k_list = np.random.permutation(len(sequences))
        for k in random_k_list:
            A[k] = sample_Ak(sequences[k], Theta, u, P, W)

        Theta = sample_Theta(sequences, A, W, alpha)
        ll = log_likelihood(sequences, A, Theta, u, P, W)
        if ll > best_ll:
            best_ll = ll
            best_A = A[:]
            best_Theta = Theta.copy()
        lls.append(ll)

        if n % shift_freq == 0:
            for delta in [-2, -1, 1, 2]:
                A, Theta = shift(sequences, A, Theta, u, P, W, delta, alpha)
                ll = log_likelihood(sequences, A, Theta, u, P, W)
                if ll > best_ll:
                    best_ll = ll
                    best_A = A[:]
                    best_Theta = Theta.copy()
                lls.append(ll)
        
    return best_A, best_Theta, lls


# Score de consensus (ScoreC)
def consensus_score(Theta):
    W = Theta.shape[1]
    score = 0.0
    for j in range(W):
        for i in range(4):
            t = Theta[i, j]
            if t > 0:
                score += t * np.log2(t)
    return 2 + score / W


# AJI et PPV
def AJI_PPV(true_A, estimated_A, W):
    n = len(true_A)
    jaccards = np.zeros(n)
    for i in range(n):
        intersection = max(0, min(true_A[i] + W - 1, estimated_A[i] + W - 1) - max(true_A[i], estimated_A[i]) + 1)
        union = max(true_A[i] + W - 1, estimated_A[i] + W - 1) - min(true_A[i], estimated_A[i]) + 1
        jaccards[i] = intersection / union if union > 0 else 0.0

    aji = np.mean(jaccards)
    ppv = np.mean([j >= 0.5 for j in jaccards])
    return aji, ppv

# Génération de séquences
def generate_sequences(N, L, W, Theta_true, u, P):
    sequences = []
    true_A = np.zeros(N, dtype=int)
    
    for k in range(N):
        seq = np.zeros(L, dtype=int)
        ak = np.random.randint(0, L - W + 1)
        true_A[k] = ak
        
        # Partie avant le motif
        if ak > 0:
            seq[0] = np.random.choice(4, p=u)
            for t in range(1, ak):
                seq[t] = np.random.choice(4, p=P[seq[t-1]])
        
        # Motif
        for j in range(W):
            seq[ak + j] = np.random.choice(4, p=Theta_true[:, j])
        
        # Partie après le motif
        if ak + W < L:
            seq[ak + W] = np.random.choice(4, p=P[seq[ak + W - 1]])
            for t in range(ak + W + 1, L):
                seq[t] = np.random.choice(4, p=P[seq[t-1]])
        
        sequences.append(seq)
    
    return sequences, true_A


#===================================================================================================================================

nuc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def run_best_of_n(sequences, W, nb_iter, alpha=1.0, shift_freq=10, n_runs=10):
    u, P = estimate_Phi(sequences)
    
    best_overall_ll = -np.inf
    best_A = None
    best_Theta = None
    all_lls = []

    for i in range(n_runs):
        A, Theta, lls = gibbs_with_shift(sequences, W, nb_iter, alpha, shift_freq)
        ll = log_likelihood(sequences, A, Theta, u, P, W)
        all_lls.append(lls)
        
        if ll > best_overall_ll:
            best_overall_ll = ll
            best_A = A.copy()
            best_Theta = Theta.copy()
        
        print(f"  Run {i+1}/{n_runs} - ll={ll:.2f} {'← meilleur' if ll == best_overall_ll else ''}")
    
    return best_A, best_Theta, all_lls


def run_myseq1():
    W=10

    Theta_true = np.ones((4, W)) * 0.05
    for j in range(W):
        dominant = np.random.randint(4)
        Theta_true[dominant, j] = 0.85
    """Theta_true = np.array([
        np.array([0.01, 0.05, 0.9, 0.6, 0.3, 0.05, 0.03, 0.01, 0.01, 0.01]),
        np.array([0.01, 0.15, 0.05, 0.1, 0.2, 0.05, 0.15, 0.01, 0.01, 0.01]),
        np.array([0.03, 0.05, 0.04, 0.1, 0.3, 0.1, 0.7, 0.03, 0.03, 0.03]),
        np.array([0.95, 0.75, 0.01, 0.2, 0.2, 0.8, 0.12, 0.95, 0.95, 0.95])
    ])"""

    u = np.ones(4) / 4
    P = np.ones((4, 4)) / 4
    """P = np.array([
        np.array([0.2, 0.2, 0.2, 0.2]),
        np.array([0.3, 0.3, 0.3, 0.3]),
        np.array([0.2, 0.2, 0.2, 0.2]),
        np.array([0.3, 0.3, 0.3, 0.3])
    ])
    P = np.transpose(P)"""

    sequences, true_A = generate_sequences(N=50, L=100, W=W, Theta_true=Theta_true, u=u, P=P)
    best_A, best_Theta, lls = gibbs_with_shift(sequences, W, nb_iter=2000)
    aji, ppv = AJI_PPV(true_A, best_A, W=W)
    print(f"AJI={aji:.4f}, PPV={ppv:.4f}")

def run_artif_seq1():
    print()
    print("=" * 60)
    print("Expérience A : séquences artificielles")
    print("=" * 60)

    seqs_artif, names_artif = load_sequences("sequences_artif.txt")
    seqs_artif_enc = []
    for seq in seqs_artif:
        seqs_artif_enc.append(np.array([nuc[c] for c in seq]))
    pos_strings, _ = load_sequences("artif_start_positions.txt")
    pattern_loc_corr = np.array([int(p) for p in pos_strings])
    W = 10
    nb_iter = 1000

    """
    pattern_loc, Theta, lls = gibbs(seqs_artif_enc, W, nb_iter)
    aji, ppv = AJI_PPV(pattern_loc_corr, pattern_loc, W)

    print(f"AJI  = {aji:.4f}")
    print(f"PPV  = {ppv:.4f}")
    print(f"Score consensus = {consensus_score(Theta):.4f}")
    print(f"Log-vraisemblance finale = {lls[-1]:.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(lls)
    plt.xlabel("Itération")
    plt.ylabel("Log-vraisemblance")
    plt.title("Convergence de l'algorithme (séquences artificielles)")
    plt.tight_layout()
    plt.savefig("Graphs/convergence_artif1.pdf")
    plt.close()
    """

    pattern_loc, Theta, lls = gibbs_with_shift(seqs_artif_enc, W, nb_iter)
    aji, ppv = AJI_PPV(pattern_loc_corr, pattern_loc, W)

    u, P = estimate_Phi(seqs_artif_enc)

    print(f"AJI  = {aji:.4f}")
    print(f"PPV  = {ppv:.4f}")
    print(f"Score consensus = {consensus_score(Theta):.4f}")
    print(f"Log-vraisemblance finale = {lls[-1]:.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(lls)
    plt.xlabel("Itération")
    plt.ylabel("Log-vraisemblance")
    plt.title("Convergence de l'algorithme (séquences artificielles)")
    plt.tight_layout()
    plt.savefig("Graphs/convergence_artif2.pdf")
    plt.close()

def run_artif_seq2():
    print()
    print("=" * 60)
    print("Expérience B : impact du nombre d'itérations")
    print("=" * 60)
    seqs_artif, names_artif = load_sequences("sequences_artif.txt")
    seqs_artif_enc = []
    for seq in seqs_artif:
        seqs_artif_enc.append(np.array([nuc[c] for c in seq]))
    pos_strings, _ = load_sequences("artif_start_positions.txt")
    pattern_loc_corr = np.array([int(p) for p in pos_strings])
    W = 10
    alpha = 1.0
    iter_values = [200, 500, 1000, 2000, 5000]

    ajis, ppvs = [], []
    for nb_iter in iter_values:
        pattern_loc, _, _ = gibbs(seqs_artif_enc, W, nb_iter, alpha)
        a, p = AJI_PPV(pattern_loc_corr, pattern_loc, W)
        ajis.append(a)
        ppvs.append(p)
        print(f"  n_iter={nb_iter:5d} -> AJI={a:.4f}, PPV={p:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(iter_values, ajis, marker="o", label="AJI")
    plt.plot(iter_values, ppvs, marker="s", label="PPV")
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Score")
    plt.title("Impact du nombre d'itérations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Graphs/impact_iterations.pdf")
    plt.close()


def run_artif_seq3():
    print()
    print("=" * 60)
    print("Expérience C : impact d'une variation de W")
    print("=" * 60)

    return 0

def run_purr_seq1():
    print()
    print("=" * 60)
    print("Expérience D : séquences PurR")
    print("=" * 60)

    seqs_purr, names_purr = load_sequences("sequences-purr.txt")
    seqs_purr_enc = [np.array([nuc[c] for c in seq]) for seq in seqs_purr]
    
    import csv
    true_positions = {}
    with open("motifs-purr.txt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['seq']
            start = int(row['start'])
            if name not in true_positions:
                true_positions[name] = start

    true_A = np.array([true_positions[name] for name in names_purr])

    W = 16
    nb_iter = 2000
    alpha = 1.0
    freq_shift = 10

    # Gibbs sans shift
    """pattern_loc, Theta, lls = gibbs(seqs_purr_enc, W, nb_iter, alpha)
    aji, ppv = AJI_PPV(true_A, pattern_loc, W)
    print("--- Gibbs sans shift ---")
    print(f"AJI  = {aji:.4f}")
    print(f"PPV  = {ppv:.4f}")
    print(f"Score consensus = {consensus_score(Theta):.4f}")
    print(f"Log-vraisemblance finale = {lls[-1]:.2f}")"""

    # Gibbs avec shift
    pattern_loc, Theta, all_lls = run_best_of_n(
        seqs_purr_enc, W, nb_iter, alpha, freq_shift, n_runs=10
    )
    aji, ppv = AJI_PPV(true_A, pattern_loc, W)
    print("--- Gibbs avec shift ---")
    print(f"AJI  = {aji:.4f}")
    print(f"PPV  = {ppv:.4f}")
    print(f"Score consensus = {consensus_score(Theta):.4f}")
    u, P = estimate_Phi(seqs_purr_enc)
    best_ll = log_likelihood(seqs_purr_enc, pattern_loc, Theta, u, P, W)
    print(f"Log-vraisemblance finale = {best_ll:.2f}")

    plt.figure(figsize=(8, 4))
    for i, lls in enumerate(all_lls):
        plt.plot(lls, alpha=0.4, label=f"Run {i+1}")
    plt.xlabel("Itération")
    plt.ylabel("Log-vraisemblance")
    plt.title("Convergence sur plusieurs runs")
    plt.tight_layout()
    plt.savefig("Graphs/convergence_purr_runs.pdf")


#===========================================================================================================

#run_myseq1()
run_artif_seq1()
#run_artif_seq2()
#run_purr_seq1()




"""
Fonctions non-optimisées et donc super lentes

# Calcul de la Log-vraisemblance pour les séquences hors motif
def log_prob_MC_seq(seq, u, P):
    if len(seq) == 0:
        return 0.0
    lp = np.log(u[seq[0]])
    for t in range(len(seq) - 1):
        lp += np.log(P[seq[t], seq[t + 1]])
    return lp


# Log-vraisemblance du motif 
def log_prob_pattern(pattern, Theta):
    lp = 0.0
    for j, n in enumerate(pattern):
        lp += np.log(Theta[n, j])
    return lp


def sample_Ak(seq, Theta, u, P, W):
    L = len(seq)
    positions = range(L - W + 1)
    log_scores = np.zeros(len(positions))
    
    for ak in range(len(positions)):
        pattern = seq[ak: ak + W]
        lp = log_prob_pattern(pattern, Theta)

        before_pattern = seq[:ak]
        if len(before_pattern) > 0:
            lp += log_prob_MC_seq(before_pattern, u, P)

        after_pattern = seq[ak + W:]
        if len(after_pattern) > 0:
            lp += np.log(P[seq[ak + W - 1], after_pattern[0]])
            for t in range(len(after_pattern) - 1):
                lp += np.log(P[after_pattern[t], after_pattern[t+1]])
        
        log_scores[ak] = lp
    
    log_scores -= np.max(log_scores)
    probs = np.exp(log_scores)
    probs /= probs.sum()
    return np.random.choice(len(positions), p=probs)

    
def log_likelihood(sequences, positions, Theta, u, P, W):
    lp = 0.0
    for k, seq in enumerate(sequences):
        ak = positions[k]
        
        before_pattern = seq[:ak]
        if len(before_pattern) > 0:
            lp += log_prob_MC_seq(before_pattern, u, P)

        pattern = seq[ak: ak + W]
        lp += log_prob_pattern(pattern, Theta)

        after_pattern = seq[ak + W:]
        if len(after_pattern) > 0:
            lp += np.log(P[seq[ak + W - 1], after_pattern[0]])
            for t in range(len(after_pattern) - 1):
                lp += np.log(P[after_pattern[t], after_pattern[t+1]])
    return lp

"""