import pkg_resources as pr

def v(n):
    try:
        print(n, pr.get_distribution(n).version)
    except:
        print(n, "NOT INSTALLED")

for n in ["fairseq", "hydra-core", "omegaconf"]:
    v(n)
