import os
import re
import subprocess
import tempfile


def bleu_score(hypotheses, references, script_dir):
    """
    Scoring function which calls the 'multi-bleu.perl' script.
    :param hypotheses: list of translation hypotheses
    :param references: list of translation references
    :param script_dir: directory containing the evaluation script
    :return: a pair (BLEU score, additional scoring information)
    """
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
        for ref in references:
            f.write(ref + '\n')

    bleu_script = os.path.join(script_dir, 'multi-bleu.perl')
    try:
        p = subprocess.Popen([bleu_script, f.name], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=open('/dev/null', 'w'))
        output, _ = p.communicate('\n'.join(hypotheses).encode("utf8"))
    finally:
        os.unlink(f.name)

    output = output.decode()

    m = re.match(r'BLEU = ([^,]*).*BP=([^,]*), ratio=([^,]*)', output)
    bleu, penalty, ratio = [float(m.group(i)) for i in range(1, 4)]

    return bleu, 'penalty={} ratio={}'.format(penalty, ratio)


def wc(filepath):
    # Count the number of file lines.
    p = subprocess.Popen(["wc", "-l", filepath], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=open("/dev/null", "w"))
    output, _ = p.communicate()
    return int(output.decode().split()[0])
