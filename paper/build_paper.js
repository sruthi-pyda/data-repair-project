/* Builds the IEEE A4 conference paper. Geometry copied from
   conference-template-a4.docx: A4 595.30x841.90pt, body margins
   top 54pt / sides 45.35pt / bottom 72pt, two columns with 18pt gutter,
   and a three-column author block (one column per author). */
const d = require("docx");
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, ShadingType, SectionType,
  PageNumber, Footer,
} = d;

const PT = 20;                                   // 1pt = 20 DXA
const PG = { width: 11906, height: 16838 };
const MAR = { top: 54 * PT, right: 45.35 * PT, bottom: 72 * PT, left: 45.35 * PT,
              header: 36 * PT, footer: 36 * PT };
const COLW = Math.floor((PG.width - MAR.left - MAR.right - 18 * PT) / 2);
const TBLW = COLW - 40;

const FONT = "Times New Roman";
const run = (t, o = {}) => new TextRun({ text: t, font: FONT, ...o });

/* ---------- paragraph helpers ---------- */
const body = (text, o = {}) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 0, line: 240 },
  indent: { firstLine: 200 },
  children: [run(text, { size: 20 })],
  ...o,
});
const H1 = (n, t) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 180, after: 90 },
  children: [run(`${n}. ${t}`, { size: 20, allCaps: true })],
});
const H2 = (l, t) => new Paragraph({
  spacing: { before: 120, after: 60 },
  children: [run(`${l}. ${t}`, { size: 20, italics: true })],
});
const caption = (t) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 80, after: 60 },
  children: [run(t, { size: 16 })],
});

/* ---------- table helper ---------- */
function table(headers, rows, widths) {
  const w = widths || headers.map(() => Math.floor(TBLW / headers.length));
  const cell = (txt, i, head) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: "E8E8E8" } : undefined,
    margins: { top: 20, bottom: 20, left: 50, right: 50 },
    children: [new Paragraph({
      alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
      spacing: { after: 0 },
      children: [run(String(txt), { size: 15, bold: !!head })],
    })],
  });
  return new Table({
    columnWidths: w,
    width: { size: w.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    borders: {
      top:    { style: BorderStyle.SINGLE, size: 4 },
      bottom: { style: BorderStyle.SINGLE, size: 4 },
      left:   { style: BorderStyle.NONE },
      right:  { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2 },
      insideVertical:   { style: BorderStyle.NONE },
    },
    rows: [
      new TableRow({ tableHeader: true,
        children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(c, i, false)) })),
    ],
  });
}

/* ---------- author block ---------- */
const AUTHORS = [
  ["1st Saanvi Sharma", "23BDS0064"],
  ["2nd Sruthi Pyda", "23BDS0176"],
  ["3rd Ruchika Tuteja", "23BDS0050"],
];
const authorCol = ([name, reg]) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
    children: [run(name, { size: 22 })] }),
  ...["School of Computer Science and Engineering",
      "Vellore Institute of Technology",
      "Vellore, Tamil Nadu, India", reg]
    .map(t => new Paragraph({ alignment: AlignmentType.CENTER,
      spacing: { after: 0 }, children: [run(t, { size: 20, italics: t === reg ? false : true })] })),
  new Paragraph({ spacing: { after: 100 }, children: [] }),
];

/* ---------- content ---------- */
const ABSTRACT =
  "Automated data repair systems are normally judged on how accurately they " +
  "correct values. We argue, and show empirically, that the harder and more " +
  "consequential problem is deciding whether to act at all. We present a " +
  "pipeline that attaches two calibrated probabilities to every flagged cell " +
  "— the probability that a proposed repair is correct, and the probability " +
  "that the cell is genuinely corrupted — and selects among three actions " +
  "(automatic repair, human review, abstention) by minimising expected cost. " +
  "Evaluated on total harm, which prices undetected corruption rather than " +
  "hiding it, the policy is best or tied-best in 74 of 75 cost " +
  "configurations across three datasets, against six alternatives including " +
  "fixed rules and confidence thresholds. An ablation isolates the " +
  "contribution of calibration: driving the identical cost rule with raw " +
  "confidence instead of the calibrated probability is never better and is " +
  "strictly worse in 29 of 75 configurations. We then report a negative " +
  "result we consider equally important: five successive mechanisms for " +
  "identifying which column is corrupted all failed, each absorbing a " +
  "different column-specific nuisance quantity rather than the corruption " +
  "signal. Using a Bayes-optimal ceiling computed from a known corruption " +
  "process we show that within a corrupted column the detector already " +
  "captures 77.9% of the attainable signal, so the residual loss is almost " +
  "entirely a column-selection problem. Finally we contribute a " +
  "string-structure fallback for columns a conditional model cannot " +
  "represent, which on two datasets yields separated improvements and on " +
  "others does not. We do not claim a competitive error detector; the " +
  "contribution is the decision layer and the measured boundary around it.";

const KEYWORDS =
  "data cleaning, error detection, probability calibration, cost-sensitive " +
  "decision making, abstention, class imbalance";

const S = [];
const P = (...xs) => S.push(...xs);

P(new Paragraph({ spacing: { after: 60 },
  children: [run("Abstract—", { size: 18, bold: true, italics: true }),
             run(ABSTRACT, { size: 18, bold: true, italics: true })],
  alignment: AlignmentType.JUSTIFIED }));
P(new Paragraph({ spacing: { after: 120 },
  children: [run("Keywords—", { size: 18, bold: true, italics: true }),
             run(KEYWORDS, { size: 18, italics: true })],
  alignment: AlignmentType.JUSTIFIED }));

/* I. INTRODUCTION */
P(H1("I", "Introduction"));
P(body("A dirty table contains cells whose values differ from the truth. The " +
  "standard formulation asks a system to find those cells and correct them, and " +
  "the standard metric rewards it for how often the correction matches the " +
  "clean value. This framing hides a question that matters more in practice: " +
  "when the system is unsure, what should it do?", { indent: { firstLine: 0 } }));
P(body("The question is not rhetorical, because the two kinds of damage are " +
  "not symmetric. Overwriting a correct value with a confident guess destroys " +
  "information. Routing every suspicious cell to a human is safe but does not " +
  "scale. Leaving an error in place is sometimes the cheapest option and " +
  "sometimes the most expensive one, depending entirely on what the data is " +
  "used for. A system that cannot express these trade-offs cannot be deployed " +
  "responsibly, however accurate its repairs are."));
P(body("We also found, repeatedly, that repair accuracy measures the dataset " +
  "rather than the method. Corruption that mangles a value — an out-of-range " +
  "number, a broken format — is recovered by simple rules 98–100% of the " +
  "time. Corruption that erases a value is recovered 29–50% of the time by " +
  "anything we tried, because the information is gone. Reporting a single repair " +
  "accuracy therefore mostly reports which mixture of the two a benchmark " +
  "happens to contain."));
P(body("This paper makes three contributions, in descending order of how well " +
  "the evidence supports them. First, a cost-sensitive decision layer that " +
  "chooses among repairing, reviewing and abstaining using two calibrated " +
  "probabilities, and which attains zero worst-case regret across the policies " +
  "and datasets we tested. Second, a measured boundary: five mechanisms for " +
  "column-level error localisation, each pre-registered with falsifying " +
  "conditions, all refuted, with a common cause we identify. Third, a targeted " +
  "fallback detector for high-cardinality text columns, which substantially " +
  "improves coverage on some datasets and not others."));

/* II. RELATED WORK */
P(H1("II", "Related Work"));
P(body("Rule-based and constraint-based cleaners such as NADEEF [1] and " +
  "HoloClean [2] detect violations of user-supplied denial constraints and " +
  "functional dependencies. Mahdavi et al. [3] run many configurations of " +
  "several base detectors, treat their outputs as a per-cell feature " +
  "vector, and learn a classifier from roughly twenty labelled tuples; a " +
  "companion system [4] handles repair. ED2 [5] adds active learning, " +
  "including a component that selects which column to label next. " +
  "HoloDetect [6] learns a corruption " +
  "policy from the dirty table itself and trains a detector on synthesised " +
  "errors, reporting that a learned augmentation policy substantially " +
  "outperforms random transformations.", { indent: { firstLine: 0 } }));
P(body("Two lines of work are closest to ours and bound our novelty claims. " +
  "HoloDetect already calibrates detector scores using Platt scaling, so " +
  "calibrating a detector is not itself new. Conformal Data Cleaning [7] pairs " +
  "per-column conditional models with calibrated uncertainty, so per-column " +
  "calibration is not new either. What we did not find in any of them is an " +
  "evaluation of calibration quality — no reliability diagram, Brier score or " +
  "expected calibration error appears — and none converts a calibrated score " +
  "into a cost-weighted choice among three actions. That gap is where our " +
  "contribution sits."));
P(body("For the mixture formulation in Section V-B, the closest prior work is " +
  "Kubica and Moore [8], who model a dirty table as a clean model plus a noise " +
  "model with a corruption matrix fitted by alternating optimisation."));

/* III. METHOD */
P(H1("III", "Method"));
P(H2("A", "Pipeline"));
P(body("The pipeline profiles each column, flags suspicious cells with six " +
  "rule checks (missing tokens, out-of-range values, Tukey fences at k = 3, " +
  "format violations, unseen categories, stray whitespace), proposes ranked " +
  "repair candidates with probabilities, and derives three uncertainty signals: " +
  "the top candidate's confidence, agreement across repeated sampling, and the " +
  "normalised entropy of the candidate distribution.", { indent: { firstLine: 0 } }));

P(H2("B", "Two calibrated probabilities"));
P(body("Raw confidence is not a probability. We fit a calibrator on a training " +
  "split to obtain p, the probability that the top repair candidate is correct, " +
  "selecting among Platt scaling, isotonic regression and a multivariate variant " +
  "by held-out Brier score.", { indent: { firstLine: 0 } }));
P(body("A flag is also not proof. We fit a second calibrator, per detection " +
  "check, to obtain q, the probability that a cell caught by that check is " +
  "genuinely corrupted. This matters because a check can be systematically " +
  "wrong: on one dataset every cell flagged by the Tukey fence is a statistical " +
  "outlier and none is an error, so q approaches zero and the system correctly " +
  "declines to repair clean data. Where a calibrator cannot be fitted, q " +
  "defaults to 1, which reproduces the uncalibrated policy exactly and is a " +
  "conservative assumption rather than evidence."));

P(H2("C", "The decision rule"));
P(body("For costs C_wrong, C_human and C_abstain, the expected cost of each " +
  "action is", { indent: { firstLine: 0 } }));
P(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 },
  children: [run("E[repair] = (1 − p)·C_wrong,   E[review] = C_human,", { size: 20, italics: true })] }));
P(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
  children: [run("E[abstain] = q·C_abstain", { size: 20, italics: true })] }));
P(body("and the action taken is the arg min, with ties broken in the order " +
  "repair, review, abstain. C_abstain is charged only when a genuinely " +
  "corrupted cell is left unrepaired; any per-flag operational cost is " +
  "decision-invariant and is reported separately rather than folded in. The " +
  "costs are fixed constants read from a configuration file in abstract " +
  "penalty units. They are not learned, not monetary, and not supplied by the " +
  "benchmarks.", { indent: { firstLine: 0 } }));
P(body("An honest limitation of the formulation: E[review] and E[abstain] are " +
  "both constant in p, so for a single cell with fixed q one of the two is " +
  "unreachable. What q changes is which of them is live, per cell, on evidence. " +
  "All three actions being reachable is a claim about the population, not about " +
  "every cell."));

/* IV. EXPERIMENTAL SETUP */
P(H1("IV", "Experimental Setup"));
P(body("We evaluate on eleven datasets: three used throughout (Adult, Breast " +
  "Cancer, Smart Factory) and eight standard error-detection benchmarks (Beers, " +
  "Flights, Hospital, Movies, Rayyan, Restaurants, Soccer, Tax). Cell-level " +
  "prevalence ranges from 0.00097 to 0.29582, which spans three orders of " +
  "magnitude and is the regime that makes this problem hard.", { indent: { firstLine: 0 } }));
P(body("Ground truth may be used only after predictions exist. The two " +
  "calibrators consume 0/1 labels on a training split; all evaluation reads " +
  "clean data after the fact. No detection, repair or decision function " +
  "receives clean data, and automated tests assert this."));
P(body("One further control deserves emphasis because it changed our " +
  "conclusions. The column profile that supplies valid ranges, vocabularies and " +
  "fences was originally built from clean data. We ablated this by rebuilding " +
  "every profile statistic from the dirty table alone. The effect is reported " +
  "in Section V-D and is large, so unless stated otherwise all detector numbers " +
  "in this paper are the zero-knowledge ones."));
P(body("Because five of our eleven datasets fall at or below the row cap used " +
  "for subsampling, repeated row splits are not independent replicates on those " +
  "datasets. We therefore vary the model seed across replicates and additionally " +
  "report a 95% bootstrap confidence interval computed by resampling evaluated " +
  "cells, which remains valid for a single run of a small table. The " +
  "intervals in Table IV are of that kind, and overlapping intervals are " +
  "reported as overlapping rather than as improvements."));

/* V. RESULTS */
P(H1("V", "Results"));

P(H2("A", "Cost-sensitive decisions"));
P(body("Every policy is scored on total harm, which charges wrong " +
  "auto-repairs, human reviews, and every corrupted cell left unrepaired " +
  "— including corruption the detector never flagged. That last term is " +
  "what prevents a policy that detects nothing from appearing cheap, and we " +
  "carry a do-nothing reference line for the same reason.", { indent: { firstLine: 0 } }));
P(body("Because a single cost setting proves little, we sweep the two ratios " +
  "that matter. Fixing C_human = 1, C_wrong takes values in {2, 5, 10, 25, " +
  "50} and C_abstain in {0.5, 1, 2, 5, 10}, giving 25 configurations per " +
  "dataset and 75 in total. Table I counts how often each policy is best or " +
  "tied-best."));
P(caption("TABLE I.	TIMES BEST OR TIED-BEST ON TOTAL HARM, OF 25 COST CONFIGURATIONS PER DATASET"));
P(table(
  ["Policy", "Breast C.", "Smart F.", "Adult", "All"],
  [["Proposed", "24/25", "25/25", "25/25", "74/75"],
   ["Uncalibrated", "0/25", "21/25", "25/25", "46/75"],
   ["Always abstain", "5/25", "9/25", "25/25", "39/75"],
   ["Always review", "8/25", "16/25", "0/25", "24/75"],
   ["Confidence ≥ 0.70", "0/25", "16/25", "0/25", "16/75"],
   ["Confidence ≥ 0.90", "0/25", "16/25", "0/25", "16/75"],
   ["Always repair", "1/25", "0/25", "0/25", "1/75"]],
  [1400, 900, 900, 800, 800]));
P(body("The proposed policy is best or tied in 74 of 75 configurations. It " +
  "is not universally best: on Breast Cancer one configuration favours " +
  "always-repair, and we report that rather than restricting the grid. Every " +
  "fixed rule is adequate in some region of the cost space and badly beaten " +
  "elsewhere. Several policies can tie on a configuration, so the columns do " +
  "not sum to 25.", { indent: { firstLine: 0 } }));
P(body("The claim is bounded by what was tested: three datasets, seven " +
  "candidate policies, this cost grid and this action space. It is not a " +
  "claim of optimality, and a different action space or a cost structure " +
  "outside the swept ratios could change it."));
P(body("One result qualifies it sharply. On Adult the proposed policy " +
  "achieves total harm 8,505, which is exactly what doing nothing achieves. " +
  "It avoids between 11,658 and 58,290 units relative to every active " +
  "baseline, and adds nothing over having no detector at all, because " +
  "Adult's 1,701 genuine errors are never flagged and so enter every policy " +
  "identically. Reporting flagged-cell cost alone would have concealed this."));
P(H2("B", "Does calibration change the decision?"));
P(body("The argument for calibration is only worth making if calibrated " +
  "probabilities select different and better actions than raw confidence. We " +
  "test this directly by driving the identical cost rule with raw confidence " +
  "in place of the calibrated p, holding the detector, the candidates, q and " +
  "the costs fixed, so any difference is attributable to calibration alone.", { indent: { firstLine: 0 } }));
P(caption("TABLE II.	CALIBRATED AGAINST RAW CONFIDENCE, 75 COST CONFIGURATIONS"));
P(table(
  ["Outcome", "Configurations"],
  [["Calibrated strictly better", "29/75"],
   ["Raw confidence strictly better", "0/75"],
   ["Identical decisions", "46/75"]],
  [2600, 1600]));
P(body("Calibration never produces a worse outcome and produces a strictly " +
  "better one in 29 of 75 configurations, saving a mean of 183.0 harm units " +
  "and up to 3,188. The benefit is therefore real but concentrated: in 46 " +
  "configurations the two agree exactly, because the cost comparison there " +
  "is decided by quantities the calibration map preserves. On Breast Cancer, " +
  "where calibration moves probabilities most, the uncalibrated policy is " +
  "best in none of its 25 configurations against the calibrated policy's 24."));

P(H2("C", "A measured boundary of detection"));
P(body("Our detector ranks cells well inside a column and poorly across " +
  "columns. Scoring only the corrupted column of Adult gives PR-AUC 0.6873; " +
  "scoring all fifteen gives 0.0123; zeroing the fourteen clean columns returns " +
  "exactly 0.6873. The equality shows the loss is displacement, not dilution — " +
  "rare but correct values in other columns outrank genuine errors.", { indent: { firstLine: 0 } }));
P(body("We built five mechanisms to select the corrupted column, each " +
  "pre-registered with conditions that could falsify it, and each was refuted. " +
  "Table III records what each one measured instead."));
P(caption("TABLE III.	REFUTED MECHANISMS AND THE QUANTITY EACH ABSORBED"));
P(table(
  ["Mechanism", "Nuisance absorbed"],
  [["Likelihood ratio", "background density"],
   ["Self-supervised injection", "column cardinality"],
   ["Contamination fraction", "model competence"],
   ["Contamination mixture", "model error rate"],
   ["Excess surprise", "model miscalibration"]],
  [2000, 2400]));
P(body("The common cause is that every per-column statistic we constructed " +
  "absorbed a column-specific nuisance quantity larger than the corruption " +
  "signal. Each fix removed one nuisance and exposed the next. The mixture " +
  "formulation is instructive: identifying corruption by its independence from " +
  "the rest of the row is sound in principle, but on Adult it recovered a " +
  "contamination rate of 0.1852 against a true rate of 0.0400, because 0.1852 " +
  "is approximately the conditional model's own error rate on that column. To " +
  "an imperfect model, its own mistakes also look row-independent.", { indent: { firstLine: 0 } }));
P(body("Because the corruption process is known for injected data, the " +
  "Bayes-optimal posterior can be computed and gives an attainable ceiling. " +
  "Within Adult's corrupted column that ceiling is PR-AUC 0.7222 and our " +
  "detector reaches 0.5626, capturing 77.9%. Per-cell ranking inside a column " +
  "is therefore close to exhausted, and essentially all remaining loss is " +
  "column selection. Measuring what perfect column selection would be worth, " +
  "however, shows it changes the whole-table result by 69.4× on Adult and only " +
  "1.0–1.7× on the other ten datasets, because those have corruption in most " +
  "of their columns. Column selection is thus a far narrower problem than we " +
  "assumed when we began."));

P(H2("D", "Targeted text fallback"));
P(body("The conditional model treats a column as a categorical variable and " +
  "skips high-cardinality text, which on real benchmarks is most of the table: " +
  "it models 0 of 7 columns on Flights, 2 of 17 on Movies and 8 of 20 on " +
  "Hospital. We add a detector that models the string rather than the category, " +
  "combining shape rarity, character n-gram implausibility, and — the " +
  "discriminating signal — similarity to the nearest frequent value. A typo " +
  "sits close to a common value without equalling it, whereas a legitimately " +
  "rare value sits far from everything. This inverts the usual rarity heuristic " +
  "and is what separates the two cases.", { indent: { firstLine: 0 } }));
P(caption("TABLE IV.	WHOLE-TABLE PR-AUC WITH 95% BOOTSTRAP CI, BASE AND WITH FALLBACK"));
P(table(
  ["Dataset", "Base", "+ Fallback"],
  [["Flights", "0.2958 [.289,.302]", "0.8360 [.828,.844]"],
   ["Hospital", "0.2632 [.225,.303]", "0.4974 [.454,.542]"],
   ["Beers", "0.4104 [.395,.427]", "0.5087 [.492,.526]"],
   ["Rayyan", "0.0931 [.086,.100]", "0.1639 [.152,.177]"],
   ["Tax", "0.0739 [.014,.160]", "0.0966 [.024,.192]"],
   ["Movies", "0.0872 [.081,.094]", "0.0876 [.082,.094]"]],
  [1400, 1500, 1500]));
P(body("Flights is the clearest case: the conditional model scores no columns " +
  "at all there, so its 0.2958 is simply the prevalence. On Flights and " +
  "Hospital the intervals before and after are disjoint, so those gains are " +
  "separated rather than merely larger point estimates. Elsewhere the picture " +
  "is weaker and the scope must be stated plainly. On Tax and Movies the " +
  "intervals overlap, so no improvement is claimed there; on two datasets a " +
  "pure rarity baseline matches or beats the fallback; and the detector must " +
  "be gated to abstain on columns where it cannot discriminate, because it " +
  "otherwise scores columns containing no errors at all and displaces true " +
  "positives in the ranking. This is a targeted extension for text-heavy " +
  "tables, not a general detector.", { indent: { firstLine: 0 } }));

P(H2("E", "What the detector depends on"));
P(body("Removing clean-derived profile statistics is severe for rule-based " +
  "detection: Breast Cancer PR-AUC falls from 0.6712 to 0.5224 with " +
  "non-overlapping intervals, and Hospital F1 falls from 0.9271 to 0.0000, " +
  "because the novel-category and valid-range checks are by definition " +
  "comparisons against a known-good vocabulary. Build that vocabulary from " +
  "dirty data and the corrupted values are already inside it.", { indent: { firstLine: 0 } }));
P(body("We therefore make no claim about competitive detection accuracy. " +
  "The detector exists to supply the decision layer with a calibrated " +
  "probability, and the evidence in this paper concerns the decisions " +
  "rather than the ranking.", { indent: { firstLine: 0 } }));

/* VI. LIMITATIONS */
P(H1("VI", "Limitations"));
P(body("We state these as findings rather than caveats. The policy is best " +
  "or tied in 74 of 75 cost configurations, not all of them: on Breast " +
  "Cancer one configuration favours always-repair. The result is relative " +
  "to three datasets, seven candidate policies and this action space, and " +
  "it is not a claim of optimality. On Adult the system equals doing " +
  "nothing once undetected errors are priced, because its 1,701 genuine " +
  "errors are never flagged and enter every policy identically. " +
  "Calibration leaves 46 of 75 configurations unchanged, so its benefit is " +
  "real but concentrated rather than uniform. One result is unexplained: " +
  "on Rayyan the combined system improves whole-table PR-AUC by 0.0708 " +
  "although the single newly covered column scores below chance, and we do " +
  "not yet know why. Five of our eleven datasets are too small for " +
  "independent row splits, so uncertainty there rests on a cell-level " +
  "bootstrap rather than resampled runs. Finally, the costs are " +
  "assumptions: we show that decisions change when they change, not that " +
  "any particular setting is correct.", { indent: { firstLine: 0 } }));

/* VII. CONCLUSION */
P(H1("VII", "Conclusion"));
P(body("Deciding whether to act is a better-posed problem than deciding what " +
  "to write. A calibrated probability of correctness, a calibrated probability " +
  "of corruption, and explicit costs give a policy that is never badly beaten " +
  "on the datasets we tested, and that survives the removal of every " +
  "clean-data advantage. The detection side of the same system is bounded in a " +
  "way we can now describe precisely: inside a column it captures 77.9% of the " +
  "attainable signal, and across columns it is defeated by nuisance quantities " +
  "that five successive mechanisms each absorbed in turn. We think the measured " +
  "boundary is as useful a contribution as the policy, and considerably more " +
  "useful than another tuned detector.", { indent: { firstLine: 0 } }));

/* REFERENCES */
P(H1("", "References"));
const REFS = [
  "M. Dallachiesa et al., “NADEEF: A commodity data cleaning system,” in Proc. ACM SIGMOD, 2013, pp. 541–552.",
  "T. Rekatsinas, X. Chu, I. F. Ilyas, and C. Ré, “HoloClean: Holistic data repairs with probabilistic inference,” PVLDB, vol. 10, no. 11, pp. 1190–1201, 2017.",
  "M. Mahdavi et al., “Raha: A configuration-free error detection system,” in Proc. ACM SIGMOD, 2019, pp. 865–882.",
  "M. Mahdavi and Z. Abedjan, “Baran: Effective error correction via a unified context representation,” PVLDB, vol. 13, no. 12, pp. 1948–1961, 2020.",
  "F. Neutatz, M. Mahdavi, and Z. Abedjan, “ED2: A case for active learning in error detection,” in Proc. ACM CIKM, 2019, pp. 2249–2252.",
  "A. Heidari, J. McGrath, I. F. Ilyas, and T. Rekatsinas, “HoloDetect: Few-shot learning for error detection,” in Proc. ACM SIGMOD, 2019, pp. 829–846.",
  "S. Jäger and F. Bießmann, “From data imputation to data cleaning: Automated cleaning of tabular data improves downstream predictive performance,” in Proc. AISTATS, PMLR vol. 238, 2024.",
  "J. Kubica and A. Moore, “Probabilistic noise identification and data cleaning,” in Proc. IEEE ICDM, 2003, pp. 131–138.",
];
REFS.forEach((r, i) => P(new Paragraph({
  spacing: { after: 20 }, alignment: AlignmentType.JUSTIFIED,
  indent: { left: 260, hanging: 260 },
  children: [run(`[${i + 1}]\t${r}`, { size: 16 })],
})));

/* ---------- document ---------- */
const doc = new Document({
  creator: "Saanvi Sharma, Sruthi Pyda, Ruchika Tuteja",
  title: "Calibrated Uncertainty and Explicit Costs for Action Selection in Tabular Data Cleaning",
  sections: [
    { /* title: one column */
      properties: { page: { size: PG, margin: { ...MAR, top: 27 * PT } }, column: { count: 1 } },
      children: [
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
          children: [run("Calibrated Uncertainty and Explicit Costs for Action Selection in Tabular Data Cleaning",
            { size: 48 })] }),
      ],
    },
    { /* authors: three columns, one per author */
      properties: { type: SectionType.CONTINUOUS,
        page: { size: PG, margin: { ...MAR, top: 22.5 * PT } },
        column: { count: 3, space: 36 * PT } },
      children: AUTHORS.flatMap(authorCol),
    },
    { /* body: two columns */
      properties: { type: SectionType.CONTINUOUS,
        page: { size: PG, margin: MAR },
        column: { count: 2, space: 18 * PT } },
      footers: { default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("", { size: 16 }),
                   new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] }) },
      children: S,
    },
  ],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("paper/Calibrated_Costs_Action_Selection.docx", b);
  console.log("wrote paper/Calibrated_Costs_Action_Selection.docx", b.length, "bytes");
});
