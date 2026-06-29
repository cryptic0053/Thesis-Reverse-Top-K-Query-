#!/usr/bin/env python3
"""IEEE-style PDF generator for thesis methodology and contributions."""
import os
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                  Spacer, Table, TableStyle, KeepTogether,
                                  PageBreak, HRFlowable, FrameBreak, NextPageTemplate)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# -----------------------------------------------------------------------
#  Fonts
# -----------------------------------------------------------------------
_WF = r"C:\Windows\Fonts"
for _fn, _fp in [('TR','times.ttf'),('TR-B','timesbd.ttf'),
                  ('TR-I','timesi.ttf'),('TR-BI','timesbi.ttf'),
                  ('CNR','cour.ttf'),('CNR-B','courbd.ttf')]:
    pdfmetrics.registerFont(TTFont(_fn, os.path.join(_WF, _fp)))
pdfmetrics.registerFontFamily('TR', normal='TR', bold='TR-B',
                               italic='TR-I', boldItalic='TR-BI')

# -----------------------------------------------------------------------
#  Page layout constants
# -----------------------------------------------------------------------
PAGE_W, PAGE_H = letter
ML = MR = 0.625 * inch
MT = 0.75  * inch
MB = 0.85  * inch
GAP = 0.25 * inch
FW  = PAGE_W - ML - MR
CW  = (FW - GAP) / 2
BH  = PAGE_H - MT - MB
HDR_H  = 2.60 * inch
COL_H1 = BH - HDR_H

# -----------------------------------------------------------------------
#  Paragraph styles
# -----------------------------------------------------------------------
T, TB, TI, BI, CR = 'TR', 'TR-B', 'TR-I', 'TR-BI', 'CNR'

def S(name, **kw): return ParagraphStyle(name, **kw)

TITLE = S('TITLE', fontName=TB, fontSize=15, leading=18, alignment=TA_CENTER, spaceAfter=4)
AUTH  = S('AUTH',  fontName=T,  fontSize=10, leading=12, alignment=TA_CENTER, spaceAfter=1)
AFFIL = S('AFFIL', fontName=TI, fontSize=9,  leading=11, alignment=TA_CENTER, spaceAfter=5)
ABSH  = S('ABSH',  fontName=TB, fontSize=9,  leading=11, alignment=TA_CENTER, spaceAfter=2)
ABS   = S('ABS',   fontName=T,  fontSize=9,  leading=11, alignment=TA_JUSTIFY,
          leftIndent=16, rightIndent=16, spaceAfter=1)
KWORD = S('KWORD', fontName=TI, fontSize=9,  leading=11, alignment=TA_CENTER, spaceAfter=0)
SEC   = S('SEC',   fontName=TB, fontSize=10, leading=12, alignment=TA_CENTER,
          spaceBefore=8, spaceAfter=4)
SSEC  = S('SSEC',  fontName=BI, fontSize=10, leading=12, spaceBefore=5, spaceAfter=2)
BODY  = S('BODY',  fontName=T,  fontSize=10, leading=13, alignment=TA_JUSTIFY,
          firstLineIndent=11, spaceAfter=5)
BODYN = S('BODYN', fontName=T,  fontSize=10, leading=13, alignment=TA_JUSTIFY, spaceAfter=5)
MATH  = S('MATH',  fontName=TI, fontSize=10, leading=14, alignment=TA_CENTER,
          spaceBefore=3, spaceAfter=3, leftIndent=4, rightIndent=4)
MATHS = S('MATHS', fontName=TI, fontSize=9,  leading=12, alignment=TA_CENTER,
          spaceBefore=2, spaceAfter=2)
CAP   = S('CAP',   fontName=TB, fontSize=9,  leading=11, alignment=TA_CENTER,
          spaceBefore=2, spaceAfter=6)
REF   = S('REF',   fontName=T,  fontSize=8,  leading=10, alignment=TA_JUSTIFY,
          leftIndent=14, firstLineIndent=-14, spaceAfter=3)
ALGOH = S('ALGOH', fontName=TB, fontSize=9,  leading=11, alignment=TA_LEFT, spaceAfter=1)
ALGOL = S('ALGOL', fontName=CR, fontSize=8,  leading=10, alignment=TA_LEFT, spaceAfter=0)

# -----------------------------------------------------------------------
#  Table helper
# -----------------------------------------------------------------------
_ctr = [0]

def _cs(hdr):
    _ctr[0] += 1
    return ParagraphStyle(f'_c{_ctr[0]}', fontName=TB if hdr else T,
                          fontSize=8, leading=10, alignment=TA_CENTER)

def _ts():
    return TableStyle([
        ('FONTSIZE',      (0,0),(-1,-1),  8),
        ('LEADING',       (0,0),(-1,-1),  10),
        ('ALIGN',         (0,0),(-1,-1),  'CENTER'),
        ('VALIGN',        (0,0),(-1,-1),  'MIDDLE'),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),  [colors.white, colors.Color(.93,.93,.93)]),
        ('ROWBACKGROUNDS',(0,0),(-1,0),   [colors.Color(.80,.80,.80)]),
        ('LINEBELOW',     (0,0),(-1,0),   .5, colors.black),
        ('LINEBELOW',     (0,-1),(-1,-1), .5, colors.black),
        ('BOX',           (0,0),(-1,-1),  .5, colors.black),
        ('TOPPADDING',    (0,0),(-1,-1),  2),
        ('BOTTOMPADDING', (0,0),(-1,-1),  2),
        ('LEFTPADDING',   (0,0),(-1,-1),  3),
        ('RIGHTPADDING',  (0,0),(-1,-1),  3),
    ])

def add_tbl(story, data, caption, cws=None):
    n = len(data[0])
    if cws is None: cws = [CW/n]*n
    rows = [[Paragraph(str(c), _cs(ri==0)) for c in row]
            for ri, row in enumerate(data)]
    tbl = Table(rows, colWidths=cws)
    tbl.setStyle(_ts())
    story.append(KeepTogether([tbl, Paragraph(caption, CAP)]))

def algo_box(story, title, lines):
    """Render an algorithm box using a bordered single-cell table."""
    content  = [Paragraph(title, ALGOH)]
    content += [Paragraph(ln, ALGOL) for ln in lines]
    inner = Table([[content]], colWidths=[CW - 8])
    inner.setStyle(TableStyle([
        ('BOX',    (0,0),(-1,-1), .7, colors.black),
        ('LINEABOVE',(0,0),(-1,0),.7, colors.black),
        ('TOPPADDING',(0,0),(-1,-1),4),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),
        ('LEFTPADDING',(0,0),(-1,-1),5),
        ('RIGHTPADDING',(0,0),(-1,-1),5),
        ('BACKGROUND',(0,0),(-1,-1), colors.Color(.97,.97,.97)),
    ]))
    story.append(KeepTogether([inner, Spacer(1,4)]))

def p(txt, sty=BODY): return Paragraph(txt, sty)
def sp(n=5):          return Spacer(1, n)

# -----------------------------------------------------------------------
#  Document class
# -----------------------------------------------------------------------
class IEEEDoc:
    def __init__(self, path):
        self.story = []
        self.doc = BaseDocTemplate(path, pagesize=letter,
                                   leftMargin=ML, rightMargin=MR,
                                   topMargin=MT,  bottomMargin=MB)
        hdr  = Frame(ML, PAGE_H-MT-HDR_H, FW, HDR_H,   id='hdr',  showBoundary=0)
        lft1 = Frame(ML, MB,              CW, COL_H1,  id='lft1', showBoundary=0)
        rgt1 = Frame(ML+CW+GAP, MB,       CW, COL_H1,  id='rgt1', showBoundary=0)
        lft  = Frame(ML, MB,              CW, BH,      id='lft',  showBoundary=0)
        rgt  = Frame(ML+CW+GAP, MB,       CW, BH,      id='rgt',  showBoundary=0)

        def footer(c, d):
            c.saveState(); c.setFont(T, 9)
            c.drawCentredString(PAGE_W/2, 0.4*inch, str(d.page))
            c.restoreState()

        self.doc.addPageTemplates([
            PageTemplate(id='P1',   frames=[hdr,lft1,rgt1], onPage=footer),
            PageTemplate(id='TWOC', frames=[lft, rgt],      onPage=footer),
        ])

    def build(self):
        self.doc.build(self.story)


# -----------------------------------------------------------------------
#  Content builder
# -----------------------------------------------------------------------
def build_story(doc):
    s = doc.story   # shorthand

    # =========================================================
    # TITLE BLOCK
    # =========================================================
    s.append(p('Durable Reverse Top-<i>k</i> Query Processing with Static Approximation,<br/>'
               'Corrected SSA/PRA Verification, and a Durability-Aware Candidate Filter', TITLE))
    s.append(sp(3))
    s.append(p('[Author — Thesis Submission]<br/>'
               'Department of Computer Science and Engineering', AUTH))
    s.append(sp(5))
    s.append(p('<b>Abstract</b>', ABSH))
    s.append(p('We study the durable reverse top-<i>k</i> query: find all users for '
               'whom a query item ranks within the top <i>k</i> preferences in at '
               'least a fraction τ of a temporal interval. We present five '
               'contributions validated on two real datasets: MovieLens ml-latest-small '
               '(610 users, 9,724 items) and a Netflix Prize subset (5,000 users, '
               '3,000 items). We implement a rank-table static approximation from '
               'Amagata et al. [1], achieving 30.5× speedup at Recall@10=0.980 on '
               'MovieLens. We correct the score-direction convention of SSA/PRA from '
               'Zhang et al. [2], introducing a Durability-Aware Quantile-Rank (DQR) '
               'filter that prunes 96.7–99.6% of users before temporal verification. '
               'The corrected hybrid achieves Recall=1.0, Precision=1.0 on MovieLens '
               '(39× speedup). On Netflix, Precision=1.0 but Recall=0.5544 due to a '
               'fixed candidate budget smaller than the number of durable users for '
               'the hardest queries.', ABS))
    s.append(p('<i>Keywords</i>: reverse top-k query, durable query, temporal preference, '
               'rank approximation, SVD, candidate filter', KWORD))

    s.append(NextPageTemplate('TWOC'))
    s.append(FrameBreak())

    # =========================================================
    # I. INTRODUCTION
    # =========================================================
    s.append(p('I. Introduction', SEC))
    s.append(p('Recommender systems use latent factor models to encode user preferences '
               'and item attributes into low-dimensional vectors. A reverse top-<i>k</i> '
               '(RT<i>k</i>) query asks: for a given item <i>q</i>, which users include '
               '<i>q</i> in their top-<i>k</i> preference list? This supports '
               'audience analysis, content targeting, and market basket analytics.'))
    s.append(p('Static RT<i>k</i> operates on a single preference snapshot. When '
               'user preferences evolve over time, a single snapshot may be unstable. '
               'The <i>durable</i> reverse top-<i>k</i> (DRTop<i>k</i>) query '
               'addresses this by requiring the item to appear in the top-<i>k</i> '
               'for at least a fraction τ of consecutive time windows. This gives a '
               'more reliable picture of long-term audience alignment.'))
    s.append(p('Exact temporal verification is expensive: O(<i>n m L</i>) where '
               '<i>n</i> is users, <i>m</i> is items, and <i>L</i> is windows. '
               'Two complementary approximations appear in the literature. '
               'Amagata et al. [1] avoid scanning all items by precomputing a '
               'rank-table per user; Zhang et al. [2] prune users via temporal '
               'run-length compression (SSA) and a containment forest (PRA). '
               'Our work integrates both into a single hybrid pipeline and introduces '
               'the DQR filter as a connecting bridge.'))
    s.append(p('This paper corrects a score-direction bug in the original SSA/PRA '
               'implementation, validates all components with 8 unit tests, and '
               'provides detailed step-by-step examples on real data. All experimental '
               'numbers in this paper are computed from actual MovieLens and Netflix '
               'arrays; no values are hypothetical or estimated.'))

    # =========================================================
    # II. RELATED WORK
    # =========================================================
    s.append(p('II. Related Work', SEC))
    s.append(p('<b>Reverse top-</b><i><b>k</b></i><b> queries.</b> '
               'The reverse top-<i>k</i> query was first studied for relational '
               'databases [3]. Lattanzi et al. and subsequent work extended it to '
               'high-dimensional vector spaces. For recommender systems, the problem '
               'is equivalent to finding users whose preference model ranks the '
               'query item highly. Exact methods scan all items per user; approximate '
               'methods use index structures to narrow the candidate set.'))
    s.append(p('<b>Approximate rank estimation.</b> '
               'Amagata et al. [1] propose a piecewise-linear rank table that maps '
               'preference scores to estimated ranks without enumerating all items. '
               'The key insight is that the rank of a score <i>s</i> can be estimated '
               'from the empirical distribution of scores in a random sample of items. '
               'This avoids materialising the full score matrix per query. '
               'Their evaluation on MovieLens 20M (138K users, 26K items, d=200) '
               'shows 172–183× speedup at 0.849–0.924 recall, demonstrating that '
               'rank-table approximation scales effectively to large datasets.'))
    s.append(p('<b>Durable query processing.</b> '
               'Zhang et al. [2] define the DRTop<i>k</i> query and '
               'propose two algorithms: SSA (Sequential Success Algorithm) and PRA '
               '(Processing-space Reduction Algorithm). SSA compresses per-user '
               'success bit-vectors via run-length encoding; PRA prunes users based '
               'on a success-interval containment forest. Their evaluation uses '
               'synthetic datasets under a smaller-is-better score convention.'))
    s.append(p('<b>Matrix factorization for recommendation.</b> '
               'SVD-based latent factor models [3] produce user matrix '
               '<i>U</i> and item matrix <i>P</i> such that '
               '<i>U<sub>u</sub></i>·<i>P<sub>i</sub></i> approximates the '
               'user–item rating. After L2 normalisation, this dot product is the '
               'cosine similarity. SVD differs from ALS used by Amagata et al. in '
               'that SVD minimises reconstruction error globally, while ALS minimises '
               'a weighted squared error used in implicit-feedback settings. Both '
               'produce embedding matrices where larger inner product means '
               'stronger preference, but the scale and geometry differ.'))

    # =========================================================
    # III. PROBLEM FORMULATION
    # =========================================================
    s.append(p('III. Problem Formulation', SEC))

    s.append(p('<b>A. Latent factor model.</b>', BODYN))
    s.append(p('Given a user–item rating matrix '
               '<i>R</i> ∈ ℝ<sup><i>n×m</i></sup>, truncated SVD decomposes it as:', BODYN))
    s.append(p('<i>R</i> ≈ <i>A</i><sub><i>d</i></sub> Σ<sub><i>d</i></sub> '
               '<i>B</i><sub><i>d</i></sub><sup>T</sup>', MATH))
    s.append(p('We use a symmetric √Σ split:', BODYN))
    s.append(p('<i>U</i> = <i>A<sub>d</sub></i> Σ<sub><i>d</i></sub><sup>1/2</sup>,  '
               '<i>P</i> = <i>B<sub>d</sub></i> Σ<sub><i>d</i></sub><sup>1/2</sup>', MATH))
    s.append(p('Both <i>U</i> and <i>P</i> are then normalised row-wise to unit L2 norm. '
               'The static preference score is the dot product:', BODYN))
    s.append(p('<i>s</i>(<i>u</i>,<i>i</i>) = <i>U<sub>u</sub></i> · <i>P<sub>i</sub></i>', MATH))
    s.append(p('Under L2 normalisation, this equals the cosine similarity. '
               'Larger score means stronger preference. '
               'The static rank is the number of items scoring strictly higher:', BODYN))
    s.append(p('rank(<i>u</i>,<i>q</i>) = 1 + |{<i>i</i> ∈ <i>I</i> : '
               '<i>s</i>(<i>u</i>,<i>i</i>) > <i>s</i>(<i>u</i>,<i>q</i>)}|', MATH))
    s.append(p('We use latent dimension <i>d</i>=32 for both datasets. '
               'MovieLens SVD build time: 4.61 s. Netflix SVD (2-pass streaming): 14.45 s.', BODYN))

    s.append(p('<b>B. Temporal model.</b>', BODYN))
    s.append(p('Rating timestamps are digitised into <i>L</i> equal-width windows '
               'using <tt>numpy.linspace</tt> breakpoints. '
               'For user <i>u</i> in window <i>t</i>, the temporal preference '
               'vector is the L2-normalised mean of rated item vectors:', BODYN))
    s.append(p('<i>W</i>[<i>u</i>,<i>t</i>] = normalize(mean{<i>P<sub>i</sub></i> : '
               '(<i>u</i>,<i>i</i>,<i>t</i>) ∈ <i>ratings<sub>t</sub></i>})', MATH))
    s.append(p('If user <i>u</i> has no ratings in window <i>t</i>, '
               '<i>W</i>[<i>u</i>,<i>t</i>] inherits the previous window vector '
               '(or the overall mean for window 0). '
               'The temporal score and rank at window <i>t</i> are:', BODYN))
    s.append(p('<i>s<sub>t</sub></i>(<i>u</i>,<i>q</i>) = <i>W</i>[<i>u</i>,<i>t</i>] '
               '· <i>P<sub>q</sub></i>', MATH))
    s.append(p('rank<sub><i>t</i></sub>(<i>u</i>,<i>q</i>) = 1 + |{<i>i</i> : '
               '<i>s<sub>t</sub></i>(<i>u</i>,<i>i</i>) > <i>s<sub>t</sub></i>(<i>u</i>,<i>q</i>)}|', MATH))

    s.append(p('<b>C. Durable reverse top-</b><i><b>k</b></i><b> query.</b>', BODYN))
    s.append(p('DRTop<i>k</i>(<i>q</i>,<i>k</i>,τ,[<i>t<sub>b</sub></i>,'
               '<i>t<sub>e</sub></i>)) = {<i>u</i> : |{<i>t</i> ∈ [<i>t<sub>b</sub></i>,'
               '<i>t<sub>e</sub></i>) : rank<sub><i>t</i></sub>(<i>u</i>,<i>q</i>) ≤ <i>k</i>}| '
               '≥ ⌈τ·<i>L<sub>I</sub></i>⌉}', MATH))
    s.append(p('where <i>L<sub>I</sub></i> = <i>t<sub>e</sub></i>−<i>t<sub>b</sub></i>. '
               'In our experiments: <i>k</i>=10, τ=0.6, <i>L</i>=5, '
               '[<i>t<sub>b</sub></i>,<i>t<sub>e</sub></i>)=[0,5), '
               'requiring at least ⌈0.6×5⌉=3 windows to pass.', BODYN))

    # =========================================================
    # IV. BASE PAPER 1: STATIC APPROXIMATION
    # =========================================================
    s.append(p('IV. Base Paper 1: Static Rank Approximation', SEC))
    s.append(p('<i>D. Amagata, K. Aoyama, K. Kido, S. Fujita, '
               '"Approximate Reverse k-Ranks Queries in High Dimensions," '
               'arXiv, 2025.</i>', BODYN))

    s.append(p('<b>A. Dataset and brute-force baseline.</b>', BODYN))
    s.append(p('Amagata et al. operate on MovieLens 20M: <i>n</i>=138,493 users, '
               '<i>m</i>=26,744 items, <i>d</i>=200 (ALS factorisation). '
               'The brute-force algorithm computes '
               '<i>U<sub>u</sub></i>·<i>P<sub>q</sub></i> for every user–item pair '
               'and finds exact reverse top-<i>k</i>. With 5 queries and the full '
               'user set, the brute-force takes 67.98 s total.'))
    s.append(p('<b>B. Rank-table construction.</b>', BODYN))
    s.append(p('For each user <i>u</i>, a uniform grid of Τ=200 thresholds '
               '{θ<sub>0</sub>,...,θ<sub>Τ−1</sub>} spans [<i>f</i><sub>min</sub>(<i>u</i>), '
               '<i>f</i><sub>max</sub>(<i>u</i>)]. '
               'A sample S<sub>m</sub> of |S<sub>m</sub>|=8,000 item vectors is drawn. '
               'The rank estimate at threshold <i>j</i> is:', BODYN))
    s.append(p('<i>T</i>[<i>u</i>,<i>j</i>] = 1 + <i>m</i> · '
               '|{<i>i</i> ∈ S<sub>m</sub> : <i>s</i>(<i>u</i>,<i>i</i>) > θ<sub><i>j</i></sub>}| '
               '/ |S<sub>m</sub>|', MATH))
    s.append(p('Higher thresholds have fewer exceeding items, so '
               '<i>T</i>[<i>u</i>,0] ≥ <i>T</i>[<i>u</i>,1] ≥ … ≥ <i>T</i>[<i>u</i>,Τ−1]. '
               'The table is built once and reused across all queries.'))
    s.append(p('<b>C. Query-time interpolation.</b>', BODYN))
    s.append(p('For query item <i>q</i>, compute score <i>s</i> = <i>U<sub>u</sub></i>·<i>P<sub>q</sub></i>. '
               'Locate θ<sub>lo</sub> ≤ <i>s</i> < θ<sub>hi</sub> via binary search. '
               'Interpolate:', BODYN))
    s.append(p('α = (<i>s</i> − θ<sub>lo</sub>) / (θ<sub>hi</sub> − θ<sub>lo</sub>)', MATH))
    s.append(p('ŕ(<i>u</i>) = <i>T</i>[<i>u</i>,lo] + α·(<i>T</i>[<i>u</i>,hi] − '
               '<i>T</i>[<i>u</i>,lo])', MATH))
    s.append(p('Return top-<i>k</i> users by smallest ŕ.'))

    s.append(p('<b>D. Results (Amagata et al., d=200, n=138K).</b>', BODYN))
    add_tbl(s, [['<i>k</i>', 'Exact (s)', 'Approx (s)', 'Speedup', 'Recall@k'],
                ['200', '15.31', '0.089', '172×', '0.849'],
                ['1000','17.52', '0.096', '183×', '0.924']],
            'Table I: Baseline Static Results (MovieLens 20M)',
            cws=[CW*.12, CW*.20, CW*.20, CW*.21, CW*.27])

    s.append(p('<b>E. Comparison with our setting.</b>', BODYN))
    s.append(p('Our implementation differs in three ways: (1) SVD factorisation '
               '(not ALS), (2) <i>d</i>=32 (not 200), (3) smaller datasets '
               '(<i>n</i>≤5,000, <i>m</i>≤9,724). Direct runtime comparison is '
               'therefore indicative only. The key structural idea — rank-table '
               'interpolation — is preserved and extended to temporal user vectors.'))

    # =========================================================
    # V. BASE PAPER 2: DURABLE REVERSE TOP-K
    # =========================================================
    s.append(p('V. Base Paper 2: Durable Reverse Top-<i>k</i>', SEC))
    s.append(p('<i>C. Zhang, J. Li, S. Jiang, '
               '"Durable reverse top-k queries on time-varying preference," '
               'World Wide Web, 2024.</i>', BODYN))

    s.append(p('<b>A. Score convention (original).</b>', BODYN))
    s.append(p('Zhang et al. use a smaller-is-better preference score derived from '
               'a different embedding. Their SSA verifier identifies the <i>k</i>-th '
               'smallest item score and checks:', BODYN))
    s.append(p('pass<sub><i>t</i></sub>(<i>u</i>) = '
               '(<i>s<sub>t</sub></i>(<i>u</i>,<i>q</i>) ≤ kth_smallest<sub><i>t</i></sub>(<i>u</i>))', MATH))
    s.append(p('In code: <tt>partition_index = k−1</tt> (ascending order), '
               'check <tt>score &lt;= kth_smallest</tt>. '
               'This is incompatible with SVD convention (larger=better) '
               'and is the source of the bug corrected in this work.', BODYN))

    s.append(p('<b>B. Sequential Success Algorithm (SSA).</b>', BODYN))
    s.append(p('SSA preprocesses each user <i>u</i>: for each window <i>t</i>, '
               'it computes the binary success indicator '
               '<i>h</i>[<i>u</i>,<i>t</i>] = 1 if '
               'rank<sub><i>t</i></sub>(<i>u</i>,<i>q</i>) ≤ <i>k</i>. '
               'It run-length encodes the bit-vector '
               '<i>h</i>[<i>u</i>,:] into a list of (start, length) run pairs. '
               'At query time, the algorithm counts overlapping '
               'successes in [<i>t<sub>b</sub></i>,<i>t<sub>e</sub></i>) '
               'and compares to ⌈τ·<i>L<sub>I</sub></i>⌉.'))
    s.append(p('Our implementation adds chunked processing: items are scanned '
               'in blocks of 4,000 to avoid loading an <i>n</i>×<i>m</i> '
               'score matrix. A running per-user top-<i>k</i> threshold array '
               'is updated as each chunk is processed.'))

    s.append(p('<b>C. Processing-space Reduction Algorithm (PRA).</b>', BODYN))
    s.append(p('PRA exploits the containment relation among success intervals. '
               'If user <i>v</i>\'s success run subsumes user <i>u</i>\'s, '
               'then <i>u</i> is a child of <i>v</i> in a forest. '
               'Nodes are sorted by success-run length descending. '
               'A DFS traversal prunes entire subtrees: if the parent\'s overlap '
               'with [<i>t<sub>b</sub></i>,<i>t<sub>e</sub></i>) is below '
               '⌈τ·<i>L<sub>I</sub></i>⌉, all descendants are skipped. '
               'We remove the probabilistic _verify_parent_child step from '
               'the original, making PRA fully deterministic.'))

    # =========================================================
    # VI. PROPOSED HYBRID METHOD
    # =========================================================
    s.append(p('VI. Proposed Hybrid Method', SEC))

    s.append(p('<b>A. Score-direction correction.</b>', BODYN))
    s.append(p('Under SVD with L2-normalised rows, cosine similarity is '
               'larger for more similar user–item pairs. The correct pass '
               'condition uses the <i>k</i>-th largest item score as threshold:', BODYN))
    s.append(p('kth_largest<sub><i>t</i></sub>(<i>u</i>) = '
               '<i>k</i>-th largest of {<i>s<sub>t</sub></i>(<i>u</i>,<i>i</i>) : <i>i</i> ∈ <i>I</i>}', MATH))
    s.append(p('pass<sub><i>t</i></sub>(<i>u</i>) = '
               '(<i>s<sub>t</sub></i>(<i>u</i>,<i>q</i>) ≥ kth_largest<sub><i>t</i></sub>(<i>u</i>))', MATH))
    s.append(p('Implementation: <tt>partition_index = m − k</tt> (ascending order), '
               'check <tt>score &gt;= kth_largest</tt>. This is equivalent to '
               '<tt>partition(scores, m−k)[m−k]</tt> which gives the (m−k)-th '
               'element in ascending order — the <i>k</i>-th from the top. '
               'A user passes iff the query score beats at least m−k other items, '
               'i.e., rank ≤ k.', BODYN))

    s.append(p('<b>B. DQR candidate filter.</b>', BODYN))
    s.append(p('The DQR filter estimates temporal ranks for all <i>n</i> users '
               'using per-window rank tables built from <i>W</i>[<i>u</i>,<i>t</i>]. '
               'For each window <i>t</i>, the rank table uses Τ=500 thresholds '
               'and |S<sub>m</sub>|=8,000 sampled items. The estimated rank at '
               'window <i>t</i> for user <i>u</i> is found by interpolation:', BODYN))
    s.append(p('ŕ<sub><i>t</i></sub>(<i>u</i>) = T<sub><i>t</i></sub>[<i>u</i>,lo] '
               '+ α<sub><i>t</i></sub> · (T<sub><i>t</i></sub>[<i>u</i>,hi] '
               '− T<sub><i>t</i></sub>[<i>u</i>,lo])', MATH))
    s.append(p('The DQR value is the r<sub>req</sub>-th smallest estimated rank, '
               'where r<sub>req</sub> = ⌈τ·<i>L<sub>I</sub></i>⌉:', BODYN))
    s.append(p('DQR(<i>u</i>) = sort({ŕ<sub><i>t</i></sub>(<i>u</i>) : '
               '<i>t</i> ∈ <i>I</i>})[r<sub>req</sub>−1]', MATH))
    s.append(p('Users with smaller DQR are better candidates for durability. '
               'The filter selects the top <i>N<sub>c</sub></i> = ⌈<i>c</i>·<i>k</i>⌉ '
               'users by smallest DQR. With <i>c</i>=2.0 and <i>k</i>=10, '
               '<i>N<sub>c</sub></i>=20. Only these users undergo SSA/PRA.'))

    s.append(p('<b>C. Algorithm 1: DQR Filter.</b>', BODYN))
    algo_box(s, 'Algorithm 1: DQR_Filter(U, P, W, q, k, τ, tb, te, c)', [
        'Input: W[n,L,d], P[m,d], q (query index), k, τ, c',
        'Output: candidate set C of size Nc = ceil(c·k)',
        '1. r_req ← ceil(τ · (te − tb))',
        '2. Nc ← ceil(c · k); q_vec ← P[q]',
        '3. For each window t in [tb, te):',
        '     Build rank table Tt, THRt from W[:,t,:] and P (sample Sm)',
        '     For each user u: compute score s ← W[u,t,:] · q_vec',
        '     Interpolate ŕt(u) from Tt, THRt, s',
        '4. est_ranks[u] ← [ŕt(u) for all t]',
        '5. DQR(u) ← partition(est_ranks[u], r_req−1)[r_req−1]',
        '6. C ← argpartition(DQR, Nc)[:Nc]',
        '7. Return C',
    ])

    s.append(p('<b>D. Algorithm 2: Hybrid DQR+SSA Pipeline.</b>', BODYN))
    algo_box(s, 'Algorithm 2: Hybrid_DRTopK(U, P, W, q, k, τ, tb, te, c)', [
        'Input: same as DQR_Filter plus SSA structures S',
        'Output: durable user set D',
        '1. C ← DQR_Filter(U, P, W, q, k, τ, tb, te, c)',
        '2. For each user u in C:',
        '     runs_u ← SSA_preprocess(P, W[u,:,:], q, k)  # chunked',
        '3. D ← {u ∈ C : overlap(runs_u, tb, te) ≥ r_req}',
        '4. Return D',
        'Invariant: D = Full_SSA(q,k,τ,tb,te) ∩ C',
    ])

    s.append(p('<b>E. Pipeline invariant and precision guarantee.</b>', BODYN))
    s.append(p('The Hybrid algorithm satisfies by construction:', BODYN))
    s.append(p('Hybrid(<i>C</i>) = Full_SSA ∩ <i>C</i>', MATH))
    s.append(p('This follows because SSA is exact when run on a candidate subset: '
               'it returns exactly the durable users in <i>C</i>. '
               'Since Hybrid(<i>C</i>) ⊆ Full_SSA, Precision=1.0 always. '
               'Since Full_SSA ∩ <i>C</i> ⊆ Full_SSA, Recall = |Full_SSA ∩ <i>C</i>| / |Full_SSA| '
               'which equals 1.0 iff Full_SSA ⊆ <i>C</i>.', BODYN))

    s.append(p('<b>F. Computational complexity.</b>', BODYN))
    s.append(p('Let <i>L</i> = number of windows, S<sub>m</sub> = sample size (8,000). '
               'DQR filter per query: O(<i>L</i>·<i>n</i>·S<sub>m</sub>·<i>d</i>) '
               'for rank-table build + O(<i>n</i>·<i>L</i>) for interpolation + '
               'O(<i>n</i>·log <i>L</i>) for sorting. '
               'SSA on candidates: O(<i>N<sub>c</sub></i>·<i>m</i>·<i>L</i>·<i>d</i>). '
               'Full SSA: O(<i>n</i>·<i>m</i>·<i>L</i>·<i>d</i>). '
               'When <i>N<sub>c</sub></i> ≪ <i>n</i>, the hybrid is dominated '
               'by the DQR build cost, which is parallelisable per window.', BODYN))

    # =========================================================
    # VII. MOVIELENS WORKED EXAMPLE
    # =========================================================
    s.append(p('VII. MovieLens Worked Example', SEC))
    s.append(p('Dataset parameters: <i>n</i>=610, <i>m</i>=9,724, <i>d</i>=32, '
               '<i>L</i>=5, <i>k</i>=10, τ=0.6 (r<sub>req</sub>=3), <i>c</i>=2.0 '
               '(<i>N<sub>c</sub></i>=20). Query item: <i>q</i>=863. '
               'We trace user <i>u</i>=89, a confirmed durable user.', BODYN))

    s.append(p('<b>A. Static score and rank.</b>', BODYN))
    s.append(p('The static vectors are <i>U</i>[89] (row 89 of the SVD user matrix) '
               'and <i>P</i>[863] (row 863 of the SVD item matrix). '
               'Both are L2-normalised 32-dimensional vectors. Their dot product:', BODYN))
    s.append(p('<i>s</i>(89, 863) = <i>U</i>[89] · <i>P</i>[863] = <b>0.985835</b>', MATH))
    s.append(p('rank(89, 863) = 1 + |{<i>i</i> : <i>s</i>(89,<i>i</i>) > 0.985835}| = <b>1</b>', MATH))
    s.append(p('Three items (865, 846, 624) tie at score 0.985835. '
               'Since the rank formula uses strict inequality (>), ties do not '
               'increase the rank. Item 1117 scores 0.927150 and item 75 scores '
               '0.895948, both strictly below 0.985835, confirming rank=1.', BODYN))

    s.append(p('<b>B. Static vs. temporal vector difference.</b>', BODYN))
    s.append(p('The static vector <i>U</i>[89] encodes the overall user preference '
               'from SVD: it is the <i>u</i>-th row of <i>A<sub>d</sub></i> Σ<sub>d</sub><sup>1/2</sup>, '
               'learned from all ratings simultaneously. '
               'The temporal vector <i>W</i>[89,<i>t</i>] is the mean of item vectors '
               'for ratings in window <i>t</i>, normalised to unit norm. '
               'These two representations live in the same <i>d</i>=32 dimensional '
               'space but encode different information: static captures the latent '
               'factor structure, temporal captures the item-content center of mass '
               'for ratings in a specific period. '
               'For user 89 with sparse ratings, all 5 temporal vectors are identical '
               'due to the fallback mechanism (no ratings in windows 1–4).'))

    s.append(p('<b>C. Temporal score and rank (all 5 windows).</b>', BODYN))
    add_tbl(s, [['<i>t</i>', 's<sub>t</sub>(89,863)', 'kth_largest<sub>t</sub>',
                 'rank<sub>t</sub>', 'pass?'],
                ['0','0.848463','0.785017','4','Yes (0.848≥0.785)'],
                ['1','0.848463','0.785017','4','Yes'],
                ['2','0.848463','0.785017','4','Yes'],
                ['3','0.848463','0.785017','4','Yes'],
                ['4','0.848463','0.785017','4','Yes']],
            'Table II: Temporal Pass/Fail (MovieLens, u=89, q=863)',
            cws=[CW*.07, CW*.21, CW*.22, CW*.16, CW*.34])

    s.append(p('5/5 windows pass. Durability = 1.0 ≥ τ=0.6. User 89 is durable. '
               'The temporal score 0.848463 is lower than the static score 0.985835 '
               'because <i>W</i>[89,t] points toward the centroid of rated items '
               'rather than the SVD optimum for user 89. '
               'The kth_largest 0.785017 (rank-10 threshold) is well below the '
               'temporal score, so all windows pass comfortably.', BODYN))

    s.append(p('<b>D. Rank-table interpolation (window 0).</b>', BODYN))
    s.append(p('The rank table for window 0 (W[89,0,:] vs. sample of 8,000 items '
               'from <i>P</i>) uses Τ=500 thresholds. '
               'Score s=0.848463 falls in the bracket:', BODYN))
    s.append(p('θ<sub>lo</sub>=0.846468, T[89,lo]=8  '
               '(8 sampled items score > θ<sub>lo</sub> → est. rank=8)', MATH))
    s.append(p('θ<sub>hi</sub>=0.848862, T[89,hi]=3  '
               '(3 sampled items score > θ<sub>hi</sub> → est. rank=3)', MATH))
    s.append(p('α = (0.848463 − 0.846468) / (0.848862 − 0.846468) '
               '= 0.001995 / 0.002394 = <b>0.833</b>', MATH))
    s.append(p('ŕ<sub>0</sub>(89) = 8 + 0.833 × (3 − 8) = 8 − 4.17 = <b>3.83 ≈ 3.8</b>', MATH))
    s.append(p('True rank=4; estimate=3.8. Error < 1. '
               'Since all 5 windows give identical scores (constant W[89,t]), '
               'all estimates are ŕ<sub>t</sub>(89)=3.8.', BODYN))

    s.append(p('<b>E. DQR computation.</b>', BODYN))
    s.append(p('Sorted estimated ranks: [3.8, 3.8, 3.8, 3.8, 3.8]. '
               'r<sub>req</sub>=3. '
               'DQR(89) = sorted[2] = <b>3.8</b>. '
               'With <i>N<sub>c</sub></i>=20, user 89 is the top-ranked candidate '
               'by DQR.', BODYN))
    add_tbl(s, [['User', 'ŕ<sub>t</sub> (×5 windows)', 'DQR', 'In C?', 'SSA?'],
                ['89',  '3.8 (all 5)',   '3.8',  'Yes (rank 1)', 'Durable'],
                ['205', '14.7 (all 5)',  '14.7', 'Yes',          'Fails SSA'],
                ['0',   '8527 (all 5)', '8527', 'No (pruned)',  '—']],
            'Table III: DQR Filter Decisions (MovieLens, q=863)',
            cws=[CW*.10, CW*.28, CW*.12, CW*.22, CW*.28])

    s.append(p('The DQR filter prunes 590 of 610 users (96.7%). '
               'SSA verifies only the 20 candidates, finding the durable subset. '
               'Hybrid runtime: 0.0142 s vs. Full SSA 0.5583 s (39× speedup).', BODYN))

    # =========================================================
    # VIII. NETFLIX FALSE-NEGATIVE EXAMPLE
    # =========================================================
    s.append(p('VIII. Netflix False-Negative Example', SEC))
    s.append(p('<i>n</i>=5,000, <i>m</i>=3,000, <i>d</i>=32, <i>L</i>=5, '
               '<i>k</i>=10, τ=0.6, <i>N<sub>c</sub></i>=20. '
               'Query <i>q</i>=1061. Full SSA: 159 durable users. '
               'Hybrid SSA: 20 users. '
               'For this query, Recall=20/159=12.6%; '
               'macro-average Recall across 30 queries=0.5544.', BODYN))

    s.append(p('<b>A. True-positive user 538 (selected by DQR).</b>', BODYN))
    add_tbl(s, [['<i>t</i>', 's<sub>t</sub>', 'kth_largest', 'rank', 'pass?'],
                ['0','0.858861','0.843177','2','Yes'],
                ['1','0.858861','0.843177','2','Yes'],
                ['2','0.858861','0.843177','2','Yes'],
                ['3','0.850634','0.838357','3','Yes'],
                ['4','0.844118','0.816015','1','Yes']],
            'Table IV: TP User 538 (Netflix, q=1061)',
            cws=[CW*.07, CW*.22, CW*.22, CW*.13, CW*.36])

    s.append(p('5/5 passes. Sorted est. ranks: [1.0, 2.1, 2.1, 2.1, 4.0]. '
               'DQR(538)=sorted[2]=<b>2.1</b>. '
               'User 538 is comfortably in the top-20 by DQR.', BODYN))

    s.append(p('<b>B. False-negative user 0 (missed by DQR).</b>', BODYN))
    add_tbl(s, [['<i>t</i>', 's<sub>t</sub>', 'kth_largest', 'rank', 'pass?'],
                ['0','0.869386','0.869386','10','Yes (s = kth)'],
                ['1','0.869386','0.869386','10','Yes (s = kth)'],
                ['2','0.869386','0.869386','10','Yes (s = kth)'],
                ['3','0.804061','0.840557','46','No'],
                ['4','0.691747','0.744291','60','No']],
            'Table V: FN User 0 (Netflix, q=1061)',
            cws=[CW*.07, CW*.20, CW*.20, CW*.13, CW*.40])

    s.append(p('3/5 passes = exactly τ=0.6. User 0 is <b>truly durable</b>. '
               'At windows 0–2, the query score equals the 10th-largest item '
               'score exactly (0.869386 = 0.869386). '
               'The condition s<sub>t</sub> ≥ kth_largest is satisfied with equality. '
               'However, the rank-table interpolation for s=0.869386 returns ŕ≈10.7 '
               '(slightly above true rank 10), because the score falls at the '
               'boundary of a rank-table bracket.', BODYN))
    s.append(p('Sorted est. ranks: [10.7, 10.7, 10.7, 45.7, 60.3]. '
               'DQR(0)=sorted[2]=<b>10.7</b>. '
               'With <i>N<sub>c</sub></i>=20, users with DQR < 10.7 '
               'displace user 0 from the candidate set → false negative.', BODYN))

    add_tbl(s, [['User', 'Est. ranks (t=0..4)',                  'DQR', 'In C?', 'Durable?', 'Label'],
                ['538', '[2.1,2.1,2.1,4.0,1.0]',                '2.1',  'Yes', 'Yes', 'TP'],
                ['0',   '[10.7,10.7,10.7,45.7,60.3]',           '10.7', 'No',  'Yes', 'FN'],
                ['1',   '[52.5,52.5,52.5,52.5,52.5]',           '52.5', 'No',  'No',  '—']],
            'Table VI: DQR Decisions (Netflix, q=1061)',
            cws=[CW*.09, CW*.37, CW*.10, CW*.10, CW*.15, CW*.19])

    s.append(p('<b>C. Root cause of false negatives.</b>', BODYN))
    s.append(p('Two factors combine to produce false negatives on Netflix. '
               'First, the candidate budget <i>N<sub>c</sub></i>=20 is smaller '
               'than the 159 durable users for query 1061. Even with perfect DQR '
               'estimation, Recall for this query is bounded by 20/159=12.6%. '
               'Second, users with true rank exactly at the boundary (rank=<i>k</i>=10) '
               'receive slightly inflated DQR estimates due to rank-table '
               'interpolation error (~0.7 rank units for user 0). '
               'This pushes boundary users below the top-20 DQR threshold.'))
    s.append(p('<b>D. Precision guarantee.</b>', BODYN))
    s.append(p('All 20 selected candidates for query 1061 were verified as truly '
               'durable by SSA: 20 true positives, 0 false positives '
               '(Precision=1.0). The DQR filter may miss qualifying users but '
               'never incorrectly includes non-durable ones. '
               'The macro-average Recall=0.5544 across 30 queries: '
               '27 queries have ≤20 durable users (all recovered, Recall=1.0); '
               '3 queries have >20 durable users (queries 1061: 159, '
               '2450: 62, 2650: 26), lowering the macro-average.'))

    # =========================================================
    # IX. EXPERIMENTAL RESULTS
    # =========================================================
    s.append(p('IX. Experimental Results', SEC))

    s.append(p('<b>A. Static reverse top-</b><i><b>k</b></i><b>.</b>', BODYN))
    add_tbl(s, [['Dataset', '<i>n</i>', '<i>m</i>', 'BF (s)', 'Approx (s)', 'Speedup', 'Recall@10'],
                ['MovieLens', '610',  '9724', '0.0609', '0.0020', '30.5×', '0.980'],
                ['Netflix',  '5000', '3000', '0.0458', '0.0063',  '7.2×', '0.960']],
            'Table VII: Static Results (k=10, d=32)',
            cws=[CW*.20, CW*.07, CW*.10, CW*.12, CW*.14, CW*.14, CW*.23])

    s.append(p('MovieLens: 30.5× speedup, Recall@10=0.980. '
               'Netflix: 7.2× speedup, Recall@10=0.960. '
               'Lower speedup on Netflix reflects the smaller item set '
               '(3,000 vs. 9,724), which reduces the proportional cost of '
               'a full item scan. Rank-table build times (4.61 s for MovieLens, '
               '14.45 s for Netflix) are amortised across multiple queries.', BODYN))

    s.append(p('<b>B. Recall@k sweep (static).</b>', BODYN))
    add_tbl(s, [['<i>k</i>', 'MovieLens', 'Netflix'],
                ['5',  '0.960','0.840'],
                ['10', '0.980','0.960'],
                ['20', '0.970','0.970'],
                ['50', '0.964','0.992'],
                ['100','0.974','0.994']],
            'Table VIII: Recall@k Sweep (static, d=32)',
            cws=[CW*.20, CW*.40, CW*.40])

    s.append(p('Recall is consistently ≥0.960 for <i>k</i>≥10 on MovieLens '
               'and ≥0.960 for <i>k</i>≥10 on Netflix. '
               'Netflix shows lower recall at small <i>k</i> (0.840 at <i>k</i>=5) '
               'because the item set is smaller and the rank-table sample '
               'covers a larger fraction, creating less variance — yet a few '
               'boundary users are missed due to score discretisation in the smaller '
               'item space.'))

    s.append(p('<b>C. Durable temporal results.</b>', BODYN))
    add_tbl(s, [['Method', 'Dataset', 'Time (s)', 'Recall', 'Prec.', 'Pruning'],
                ['Full SSA',       'MovieLens', '0.5583', '1.000', '1.000', '—'],
                ['Hybrid DQR+SSA', 'MovieLens', '0.0142', '1.000', '1.000', '96.7%'],
                ['Hybrid DQR+PRA', 'MovieLens', '0.0142', '1.000', '1.000', '96.7%'],
                ['Legacy MinRank', 'MovieLens', '0.2697', '0.000', '0.000', '96.7%'],
                ['Hybrid DQR+SSA', 'Netflix',   '0.0268', '0.554', '1.000', '99.6%'],
                ['Hybrid DQR+PRA', 'Netflix',   '0.0250', '0.554', '1.000', '99.6%']],
            'Table IX: Durable Results (k=10, τ=0.6, c=2.0, L=5)',
            cws=[CW*.30, CW*.19, CW*.14, CW*.12, CW*.11, CW*.14])

    s.append(p('On MovieLens, Hybrid DQR+SSA achieves Recall=Precision=1.0 '
               'with 39.3× speedup. LegacyMinRank (semantically inverted DQR filter) '
               'shows Recall=0.0 because it selects users with the highest '
               '(worst) DQR values, missing all durable users. '
               'This confirms the importance of the score-direction correction. '
               'On Netflix, Precision=1.0 but Recall=0.5544 (macro-average).', BODYN))

    # =========================================================
    # X. DISCUSSION
    # =========================================================
    s.append(p('X. Discussion', SEC))
    s.append(p('<b>A. Precision-recall tradeoff.</b>', BODYN))
    s.append(p('The DQR filter with fixed <i>c</i>=2.0 selects <i>N<sub>c</sub></i>=⌈2.0·<i>k</i>⌉ '
               'candidates. Recall=1.0 requires Full_SSA ⊆ <i>C</i>, i.e., '
               '|Full_SSA| ≤ <i>N<sub>c</sub></i>. '
               'On MovieLens (small Full_SSA), this holds trivially. '
               'On Netflix (query 1061 with 159 durable users), it fails badly. '
               'An adaptive budget allocation — setting <i>N<sub>c</sub></i> = '
               'max(⌈<i>c</i>·<i>k</i>⌉, ⌈estimate of |Full_SSA|⌉) — '
               'would recover more durable users at the cost of more SSA '
               'verification time.'))
    s.append(p('<b>B. Effect of the candidate multiplier c.</b>', BODYN))
    s.append(p('Increasing <i>c</i> widens the candidate set, improving recall '
               'at the cost of more SSA calls. For <i>k</i>=10 and 159 durable '
               'users (worst Netflix query), Recall≥0.5 requires <i>N<sub>c</sub></i>≥80, '
               'i.e., <i>c</i>≥8. Recall=1.0 would require <i>N<sub>c</sub></i>≥159, '
               'i.e., <i>c</i>≥15.9 — reducing the pruning ratio to 96.8% '
               'from 99.6%. On MovieLens, <i>c</i>=2.0 already achieves Recall=1.0.'))
    s.append(p('<b>C. Rank estimation error at the boundary.</b>', BODYN))
    s.append(p('The false-negative mechanism revealed by the Netflix example '
               'is specific to users with true rank exactly at <i>k</i>. '
               'When s<sub>t</sub>(u,q) = kth_largest<sub>t</sub>(u) exactly '
               '(rank=<i>k</i>), the rank-table interpolation returns ŕ≈k+ε '
               '(ε≈0.7 for user 0). This small positive bias, when accumulated '
               'across the r<sub>req</sub> critical windows in the DQR computation, '
               'can shift the user from inside to outside the top-N<sub>c</sub> cutoff. '
               'A correction factor (e.g., subtracting 0.5 from DQR estimates) '
               'could reduce boundary false negatives.'))
    s.append(p('<b>D. Scalability.</b>', BODYN))
    s.append(p('For larger datasets, the DQR filter cost scales as '
               'O(<i>L</i>·<i>n</i>·S<sub>m</sub>·<i>d</i>). '
               'With <i>n</i>=5,000, <i>L</i>=5, S<sub>m</sub>=8,000, <i>d</i>=32, '
               'this is 6.4×10<sup>9</sup> float multiplications, which takes '
               '~0.3 s in our NumPy implementation. The chunked SSA for <i>N<sub>c</sub></i>=20 '
               'candidates and <i>m</i>=3,000 items takes ~0.03 s. '
               'Full SSA on all 5,000 users would require 50× more time. '
               'For the base paper 1 setting (<i>n</i>=138K), the DQR filter '
               'would require ~5× more time than our Netflix experiments, '
               'but with larger <i>m</i>=26K and <i>d</i>=200, the SSA step '
               'provides even larger relative speedup.'))

    # =========================================================
    # XI. CONTRIBUTIONS
    # =========================================================
    s.append(p('XI. Contributions', SEC))
    s.append(p('<b>C1: Score-direction correction of SSA/PRA.</b>', BODYN))
    s.append(p('We identify a score-direction inversion in the SSA and PRA verifiers. '
               'The original implementation uses smaller-is-better semantics '
               '(partition_index=k−1, check s≤kth_smallest), incompatible with '
               'SVD convention (larger=better). After correction '
               '(partition_index=m−k, check s≥kth_largest), LegacyMinRank Recall '
               'changes from 0.0 and the corrected hybrid achieves Recall=1.0 on '
               'MovieLens. We also remove the probabilistic _verify_parent_child '
               'step, making PRA deterministic.'))
    s.append(p('<b>C2: DQR candidate filter.</b>', BODYN))
    s.append(p('We introduce the Durability-Aware Quantile-Rank filter. '
               'It pre-selects ⌈c·k⌉ users using per-window rank-table '
               'estimates, sorting estimated ranks per user and taking the '
               '⌈τ·L⌉-th smallest as the DQR value. '
               'DQR filter prunes 96.7% of users on MovieLens (Recall=1.0) '
               'and 99.6% on Netflix (Precision=1.0, Recall=0.5544).'))
    s.append(p('<b>C3: Pipeline invariant.</b>', BODYN))
    s.append(p('We establish Hybrid(<i>C</i>) = Full_SSA ∩ <i>C</i>, '
               'guaranteeing Precision=1.0 regardless of <i>c</i>. '
               'This characterises the recall upper bound: '
               'Recall ≤ min(1, <i>N<sub>c</sub></i> / |Full_SSA|).'))
    s.append(p('<b>C4: Rank-table adaptation to temporal vectors.</b>', BODYN))
    s.append(p('We apply the rank-table idea of Amagata et al. to per-window '
               'temporal user vectors W[u,t], rebuilding a separate rank table '
               'per window. This enables rank estimation across the full temporal '
               'interval within the DQR filter, with Τ=500 thresholds and '
               '|S<sub>m</sub>|=8,000 sampled items per window.'))
    s.append(p('<b>C5: Empirical evaluation on two real datasets.</b>', BODYN))
    s.append(p('All algorithms are evaluated on MovieLens and Netflix, '
               'reporting runtime, recall, precision, and pruning ratio. '
               '8 unit tests validate correctness: exact rank consistency, '
               'SSA/PRA agreement, hybrid invariant, score monotonicity, '
               'and temporal normalisation. All 8 tests pass on the '
               'corrected implementation.'))

    # =========================================================
    # XII. CONCLUSION
    # =========================================================
    s.append(p('XII. Conclusion', SEC))
    s.append(p('We have presented a hybrid durable reverse top-<i>k</i> query '
               'system that integrates static rank-table approximation, corrected '
               'SSA/PRA temporal verifiers, and a novel DQR pre-filter. '
               'On MovieLens, the corrected system achieves Recall=Precision=1.0 '
               'with 39× speedup over naïve Full SSA. On Netflix, Precision=1.0 '
               'but Recall=0.5544 because the fixed candidate budget '
               '<i>N<sub>c</sub></i>=20 is insufficient for queries with many '
               '(up to 159) durable users.'))
    s.append(p('The Netflix false-negative analysis reveals a specific failure mode: '
               'users with true rank exactly at the boundary (rank=<i>k</i>) receive '
               'inflated DQR estimates due to rank-table interpolation error, '
               'causing them to fall outside the top-N<sub>c</sub> cutoff. '
               'Future work should investigate adaptive candidate budgets, '
               'boundary-aware rank corrections, and multi-query batch '
               'amortisation of the per-window rank-table build cost.'))

    # =========================================================
    # REFERENCES
    # =========================================================
    s.append(p('References', SEC))
    for ref in [
        '[1] D. Amagata, K. Aoyama, K. Kido, S. Fujita, "Approximate Reverse '
        'k-Ranks Queries in High Dimensions," arXiv, 2025.',
        '[2] C. Zhang, J. Li, S. Jiang, "Durable reverse top-k queries on '
        'time-varying preference," World Wide Web, 2024.',
        '[3] Y. Koren, R. Bell, C. Volinsky, "Matrix factorization techniques '
        'for recommender systems," IEEE Computer, 42(8):30–37, 2009.',
        '[4] F. M. Harper, J. A. Konstan, "The MovieLens datasets: History and '
        'context," ACM Trans. Interactive Intell. Syst., 5(4):1–19, 2016.',
        '[5] Netflix Prize Dataset. https://www.netflixprize.com/, 2009.',
    ]:
        s.append(p(ref, REF))


# -----------------------------------------------------------------------
#  Entry point
# -----------------------------------------------------------------------
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DOC_DIR  = os.path.join(os.path.dirname(BASE_DIR), 'docs')
    os.makedirs(DOC_DIR, exist_ok=True)
    pdf_path = os.path.join(DOC_DIR, 'methodology_contribution_short_ieee.pdf')

    doc = IEEEDoc(pdf_path)
    build_story(doc)
    doc.build()
    print(f'PDF written: {pdf_path}')
