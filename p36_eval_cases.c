#include "p36_eval_cases.h"

/* Curated hard, LifeOS and multilingual cases. Source revisions and redistribution notes are recorded in
 * EVAL_DATA.md. Keep this table data-only; grading belongs in p36_eval.c. */

const eval_source eval_hard_sources[] = {
    {"MMLU-Pro", "https://github.com/TIGER-AI-Lab/MMLU-Pro", "Apache-2.0"},
    {"OlympiadBench", "https://huggingface.co/datasets/Hothan/OlympiadBench", "Apache-2.0"},
    {"LiveBench", "https://huggingface.co/datasets/livebench/reasoning", "Apache-2.0"},
    {"NIST Juliet", "https://samate.nist.gov/SARD/test-suites/112", "CC0-1.0"},
    {"LifeOS", "EVAL_DATA.md", "MIT"},
    {"MMLU-ProX", "https://huggingface.co/datasets/li-lab/MMLU-ProX", "MIT"},
};

const size_t eval_hard_source_count =
    sizeof(eval_hard_sources) / sizeof(eval_hard_sources[0]);

const eval_case eval_hard_cases[] = {
    {
        .source = "MMLU-Pro",
        .id = "2826",
        .domain = "biology",
        .title = "Biology challenge",
        .question = "A gene C mutates to c with a frequency of 2 \303\227 10^-6 per generation. There are no othe"
        "r forces acting on these alleles and mating is random. How many generations are needed to in"
        "crease the frequency of gene c from 2 percent to 3 percent?",
        .choice[0] = "2,500 generations",
        .choice[1] = "6,200 generations",
        .choice[2] = "15,000 generations",
        .choice[3] = "8,764 generations",
        .choice[4] = "10,000 generations",
        .choice[5] = "20,000 generations",
        .choice[6] = "3,000 generations",
        .choice[7] = "1,000 generations",
        .choice[8] = "7,500 generations",
        .choice[9] = "5,128 generations",
        .answer = "J",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "86",
        .domain = "business",
        .title = "Business challenge",
        .question = "The stock of the CCC Corporation is currently valued at $12 and is assumed to possess all th"
        "e properties of geometric Brownian motion. It has an expected annual return of 15%, an annua"
        "l volatility of 20%, and the annual risk-free is 10%. Using a binomial lattice, determine th"
        "e price of a call option on CCC stock maturing in 10 monthes time with a strike price of $14"
        " (Let the distance between nodes on your tree be 1 month in length).",
        .choice[0] = "50.0",
        .choice[1] = "60.0",
        .choice[2] = "55.0",
        .choice[3] = "44.0",
        .choice[4] = "48.0",
        .choice[5] = "53.0",
        .choice[6] = "42.0",
        .choice[7] = "46.0",
        .choice[8] = "51.0",
        .choice[9] = "45.0",
        .answer = "F",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "3546",
        .domain = "chemistry",
        .title = "Chemistry challenge",
        .question = "Suppose that the normalized wavefunction for an electron in a carbon nanotube of length $L=1"
        "0.0 \\mathrm{~nm}$ is: $\\psi=(2 / L)^{1 / 2} \\sin (\\pi x / L)$. Calculate the probability"
        " that the electron is between $x=4.95 \\mathrm{~nm}$ and $5.05 \\mathrm{~nm}$.",
        .choice[0] = "0.020",
        .choice[1] = "0.050",
        .choice[2] = "0.030",
        .choice[3] = "0.045",
        .choice[4] = "0.010",
        .choice[5] = "0.015",
        .choice[6] = "0.035",
        .choice[7] = "0.060",
        .choice[8] = "0.025",
        .choice[9] = "0.040",
        .answer = "A",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "10360",
        .domain = "computer science",
        .title = "Computer Science challenge",
        .question = "Two expressions E and F are said to be unifiable if there are substitutions for the variable"
        "s of E and F that make the expressions lexically identical. In the following three expressio"
        "ns, only w, x, y, and z are variables.\nI. f(w,w)\nII. f(x,1)\nIII. f(y,g(z))\nWhich pairs o"
        "f these expressions is (are) pairs of unifiable expressions?",
        .choice[0] = "(I, III) and (II, III) only",
        .choice[1] = "(II, III) only",
        .choice[2] = "(I, II) and (I, III) only",
        .choice[3] = "(I, II) only",
        .choice[4] = "Only single expressions are unifiable, not pairs",
        .choice[5] = "None of the pairs are unifiable",
        .choice[6] = "(I, II), (I, III), and (II, III)",
        .choice[7] = "All three expressions are unifiable",
        .choice[8] = "(I, II) and (II, III) only",
        .choice[9] = "(I, III) only",
        .answer = "C",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "6844",
        .domain = "economics",
        .title = "Economics challenge",
        .question = "The purpose of \"augmenting\" the Dickey-Fuller test regression is to",
        .choice[0] = "Ensure that there is no heteroscedasticity in the test regression residuals.",
        .choice[1] = "Improve the accuracy of the Dickey-Fuller test",
        .choice[2] = "Ensure that the test regression residuals are negatively skewed",
        .choice[3] = "Ensure that the test regression residuals are normally distributed",
        .choice[4] = "Ensure that all of the non-stationarity is taken into account.",
        .choice[5] = "Ensure that the test regression residuals have no outliers",
        .choice[6] = "Ensure that the test regression residuals are positively skewed",
        .choice[7] = "Ensure that there is no autocorrelation in the test regression residuals",
        .choice[8] = "Ensure that the test regression residuals are balanced",
        .choice[9] = "Increase the complexity of the Dickey-Fuller test",
        .answer = "H",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "11292",
        .domain = "engineering",
        .title = "Engineering challenge",
        .question = "Calculate the entropy change for (a) H_2O (s, - 20\302\260C, 1atm) \\rightarrow H_2O (s, 0"
        "\302\260C, 1atm) (b) H_2O (s, - 20\302\260C, 1atm) \\rightarrow H_2O (1, 20\302\260C, 1atm) "
        "Assume C_p (H_2O, 1) = 18 cal deg^-1 mole^-1, C_p (H_2O, s) = 9 cal deg^-1 mole^-1 and\\Delt"
        "aH_fusion= 1440 cal deg^-1 mole^-1, andalso that the heat capacities are independent of temp"
        "erature.",
        .choice[0] = "0.68eu/mole for (a) and 7.22eu for (b)",
        .choice[1] = "1.27eu/mole for (a) and 7.22eu for (b)",
        .choice[2] = "0.68eu/mole for (a) and 5.27eu for (b)",
        .choice[3] = "1.27eu/mole for (a) and 5.27eu for (b)",
        .answer = "A",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "6034",
        .domain = "health",
        .title = "Health challenge",
        .question = "A 23-year-old woman with bone marrow failure is treated with a large dose of rabbit antithym"
        "ocyte globulin. Ten days later, she develops fever, lymphadenopathy, arthralgias, and erythe"
        "ma on her hands and feet. Which of the following is the most likely cause of these symptoms?",
        .choice[0] = "Activation of complement system",
        .choice[1] = "Overproduction of monocytes",
        .choice[2] = "Cytokine secretion by natural killer cells",
        .choice[3] = "Hyperactive B-lymphocyte response",
        .choice[4] = "Allergic reaction to rabbit proteins",
        .choice[5] = "Immune complex deposition in tissues",
        .choice[6] = "Polyclonal T-lymphocyte activation",
        .choice[7] = "Eosinophil degranulation",
        .choice[8] = "Increased production of interferon-gamma",
        .choice[9] = "Acute graft-versus-host disease",
        .answer = "F",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "4699",
        .domain = "history",
        .title = "History challenge",
        .question = "This question refers to the following information.\n\"At least one of the [world's] societie"
        "s would have to somehow enormously increase its productivity [in order to achieve global heg"
        "emony]. That quantum jump would have to be made before the various scientific, technological"
        ", agricultural, and industrial revolutions on which our post-quantum-leap world rests. It co"
        "uld only be accomplished by exploiting the ecosystems, mineral resources, and human assets o"
        "f whole continents outside the lands of the society making the jump. Western Europe did just"
        " that by means of its brutality and guns and, more important, by geographical and ecological"
        " luck.\"\nCopyright \302\251 2015 Cambridge University Press.\nAlfred Crosby, historian, Eco"
        "logical Imperialism, 2004\nBased on your knowledge of world history, which of the following "
        "contributed LEAST to Western European global hegemony between 1450 and 1750 C.E.?",
        .choice[0] = "The implementation of joint-stock companies",
        .choice[1] = "The exploitation of labor in colonized regions",
        .choice[2] = "The exploration of new sea routes to Asia and the Americas",
        .choice[3] = "The establishment of trading posts along Asian and African coasts",
        .choice[4] = "The exchange of food sources between the Americas and Europe",
        .choice[5] = "The spread of Christianity to non-European regions",
        .choice[6] = "The development and application of steam-powered technologies",
        .choice[7] = "The discovery and colonization of the Americas",
        .choice[8] = "The development of advanced naval technology",
        .choice[9] = "Refinement of gunpowder technologies",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "899",
        .domain = "law",
        .title = "Law challenge",
        .question = "A man and a woman were passengers on a plane. They were seated next to one another in the fi"
        "rst- class section of the plane. Midway through the flight, the woman excused herself and we"
        "nt to the restroom. While she was gone, the man rummaged through the woman's purse and stole"
        " $100 in cash. Unknown to the man, a flight attendant saw him steal the money. As is customa"
        "ry in such situations, the flight attendant radioed ahead, and police officers arrested the "
        "man when the plane landed. The man was subsequently charged with violating an applicable fed"
        "eral larceny statute. During the trial, the prosecuting attorney called the man's wife as a "
        "willing witness against her husband. She proposed to testify that the man confided to her th"
        "at he did steal the woman's money during the flight. The man's attorney objected on the grou"
        "nds of hearsay and privilege. The wife's proposed testimony is",
        .choice[0] = "admissible, because it is privileged, though it is not hearsay.",
        .choice[1] = "admissible, because it is neither hearsay nor privileged.",
        .choice[2] = "admissible, because it is hearsay within a recognized exception and not privileged.",
        .choice[3] = "inadmissible, both because it discloses a privileged communication and because it is hearsay"
        " not within any recognized exception.",
        .choice[4] = "admissible, both because it discloses a privileged communication and because it is hearsay n"
        "ot within any recognized exception.",
        .choice[5] = "inadmissible, because it is hearsay not within any recognized exception, and it discloses a "
        "privileged communication.",
        .choice[6] = "inadmissible, because it discloses a privileged communication, though it is not hearsay.",
        .choice[7] = "admissible, both because it does not disclose a privileged communication and because it is h"
        "earsay within a recognized exception.",
        .choice[8] = "inadmissible, because it is hearsay within a recognized exception, though it is not privileg"
        "ed.",
        .choice[9] = "inadmissible, because it is hearsay not within any recognized exception, though it is not pr"
        "ivileged.",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "7738",
        .domain = "math",
        .title = "Math challenge",
        .question = "Suppose that $f(x)$ is a polynomial that has degree $6$ and $g(x)$ is a polynomial that has "
        "degree $3$. If $h(x)$ is also a polynomial such that $f(g(x)) + g(h(x)) + h(f(x))$ is a poly"
        "nomial of degree $36$, then what is the degree of the polynomial $h$?",
        .choice[0] = "3",
        .choice[1] = "21",
        .choice[2] = "6",
        .choice[3] = "9",
        .choice[4] = "30",
        .choice[5] = "72",
        .choice[6] = "12",
        .choice[7] = "15",
        .choice[8] = "24",
        .choice[9] = "18",
        .answer = "C",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "5078",
        .domain = "other",
        .title = "Other challenge",
        .question = "Are strategies of Arms Control and Deterrence useful in cyber-security?",
        .choice[0] = "No, arms control does not serve any purpose outside the realm of traditional military weapon"
        "s.",
        .choice[1] = "Arms control strategies can be implemented, but deterrence strategies are impossible due to "
        "the anonymous nature of cyber attacks.",
        .choice[2] = "Yes, both strategies are effective in cyber-security as long as there is strong internationa"
        "l cooperation and regulation.",
        .choice[3] = "Effective deterrence requires international cooperation which is achievable in theory, but t"
        "he verification required for traditional arms control strategies is impossible in cyber(ed) "
        "warfare.",
        .choice[4] = "Only arms control is unproblematic, deterrence has failed because international cooperation "
        "cannot be reached and common rules or practices cannot be established.",
        .choice[5] = "Both strategies can only work if there is a clear agreement on what constitutes a cyber weap"
        "on and what constitutes a cyber attack.",
        .choice[6] = "Yes, successful policies of arms control and deterrence have translated well from other secu"
        "rity realms, such as nuclear weapons control strategies.",
        .choice[7] = "Deterrence is ineffective because it encourages an arms race in cyberspace, while arms contr"
        "ol is difficult due to the ease of creating and distributing cyber weapons.",
        .choice[8] = "Both arms control and deterrence strategies are ineffective due to the rapid and unpredictab"
        "le development of cyber warfare technologies.",
        .choice[9] = "No, both strategies are obsolete in the face of cyber warfare's unique challenges such as at"
        "tribution and boundary issues.",
        .answer = "D",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "10775",
        .domain = "philosophy",
        .title = "Philosophy challenge",
        .question = " Construct a complete truth table for the following pairs of propositions. Then, using the t"
        "ruth tables, determine whether the statements are logically equivalent or contradictory. If "
        "neither, determine whether they are consistent or inconsistent. Justify your answers.\n(~M "
        "\342\212\203 ~N) \342\210\250 (O \342\211\241 N) and (~M \302\267 N) \302\267 [(~O \342\210"
        "\250 ~N) \302\267 (O \342\210\250 N)]",
        .choice[0] = "Both logically equivalent and consistent",
        .choice[1] = "Both consistent and contradictory",
        .choice[2] = "Both logically equivalent and contradictory",
        .choice[3] = "Contradictory",
        .choice[4] = "Neither logically equivalent nor contradictory, but inconsistent",
        .choice[5] = "Logically equivalent",
        .choice[6] = "Both inconsistent and contradictory",
        .choice[7] = "Inconsistent",
        .choice[8] = "Neither logically equivalent nor contradictory, but consistent",
        .choice[9] = "Neither logically equivalent nor inconsistent, but contradictory",
        .answer = "D",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "9055",
        .domain = "physics",
        .title = "Physics challenge",
        .question = "Two spaceships approach Earth with equal speeds, as measured by an observer on Earth, but fr"
        "om opposite directions. A meterstick on one spaceship is measured to be 60 cm long by an occ"
        "upant of the other spaceship. What is the speed of each spaceship, as measured by the observ"
        "er on Earth?",
        .choice[0] = "0.4c",
        .choice[1] = "0.1c",
        .choice[2] = "1.0c",
        .choice[3] = "0.9c",
        .choice[4] = "0.2c",
        .choice[5] = "0.5c",
        .choice[6] = "0.6c",
        .choice[7] = "0.3c",
        .choice[8] = "0.8c",
        .choice[9] = "0.7c",
        .answer = "F",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "1995",
        .domain = "psychology",
        .title = "Psychology challenge",
        .question = "An examinee who makes a number of confabulatory responses on the Rorschach has overgeneraliz"
        "ed from a part of the inkblot to the whole. Such responses suggest which of the following?",
        .choice[0] = "an inability to perceive abstract concepts",
        .choice[1] = "high level of stress or anxiety",
        .choice[2] = "a low level of self-esteem",
        .choice[3] = "high intelligence and creativity",
        .choice[4] = "a lack of attention to detail",
        .choice[5] = "a need for immediate gratification",
        .choice[6] = "brain damage, intellectual disability, or emotional disturbance",
        .choice[7] = "a high degree of defensiveness or denial",
        .choice[8] = "a highly imaginative personality",
        .choice[9] = "a high degree of introversion or social withdrawal",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "7754",
        .domain = "math",
        .title = "Math challenge",
        .question = "How many triangles are there whose sides are all integers and whose maximum side length equa"
        "ls 11?",
        .choice[0] = "45",
        .choice[1] = "36",
        .choice[2] = "48",
        .choice[3] = "33",
        .choice[4] = "50",
        .choice[5] = "55",
        .choice[6] = "24",
        .choice[7] = "42",
        .choice[8] = "30",
        .choice[9] = "60",
        .answer = "B",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "3024",
        .domain = "biology",
        .title = "Biology challenge",
        .question = "Two 30\" plants are crossed, resulting in progeny of the following ratio: one 22\", eight 24"
        "\\textquotedblright, twenty-eight 26\", fifty-six 28\", seventy 30\", fifty-six 32\", twenty"
        "-eight 34\", eight 36\", and one 38\". Starting with A and going through the alphabet to rep"
        "resent different alleles, what are the most probable genotypes of the parents?",
        .choice[0] = "AaBbCcDd\303\227AABBCCDD",
        .choice[1] = "AaBbCcEeFf\303\227AaBbCcEeFf",
        .choice[2] = "AaBbCc\303\227AaBbCc",
        .choice[3] = "AaBb\303\227AaBb",
        .choice[4] = "AaBbCcDd\303\227AaBbCcDd",
        .choice[5] = "AaBbCcDdEe\303\227AaBbCcDdEe",
        .choice[6] = "Aa\303\227Aa",
        .choice[7] = "AaBbDd\303\227AaBbDd",
        .choice[8] = "AaBbCc\303\227AaBb",
        .choice[9] = "AABB\303\227AABB",
        .answer = "E",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "671",
        .domain = "business",
        .title = "Business challenge",
        .question = "Frank Bryant is a salesman for the Fast Cool Refrigerator Co. He is compensated for all trav"
        "el expenses, and is allowed a drawing account of $600 per month. He receives a monthly commi"
        "ssion of 2% of his first $30,000 of sales, 2.5% of the next $10,000 of sales, and 3% of all "
        "sales in excess of $40,000. Find the amount due him for a month in which his travel expenses"
        " paid out of pocket are $332.75, his drawings are $600, and his sales total $41,180.",
        .choice[0] = "$1,332.75",
        .choice[1] = "$942.55",
        .choice[2] = "$783.40",
        .choice[3] = "$618.15",
        .choice[4] = "$725.15",
        .choice[5] = "$1,035.40",
        .choice[6] = "$885.40",
        .choice[7] = "$1,118.15",
        .choice[8] = "$1,218.15",
        .choice[9] = "$332.75",
        .answer = "D",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "3696",
        .domain = "chemistry",
        .title = "Chemistry challenge",
        .question = "2 moles of a monoatomic ideal gas at 45\302\260C are compressed adiabatically and reversibly"
        " from 75 liters to 20 liters. Calculate q, W, \\DeltaH and \\DeltaE for the process, excludi"
        "ng vi-brationalcontributions.",
        .choice[0] = "q=0, \316\224E=15.00kJ, \316\224H=22.50kJ",
        .choice[1] = "q=11.21kJ, \316\224E=0, \316\224H=18.68kJ",
        .choice[2] = "q=0, \316\224E=18.68kJ, \316\224H=11.21kJ",
        .choice[3] = "q=18.68kJ, \316\224E=18.68kJ, \316\224H=18.68kJ",
        .choice[4] = "q=0, \316\224E=11.21kJ, \316\224H=18.68kJ",
        .choice[5] = "q=11.21kJ, \316\224E=18.68kJ, \316\224H=0",
        .choice[6] = "q=22.50kJ, \316\224E=15.00kJ, \316\224H=0",
        .choice[7] = "q=0, \316\224E=22.50kJ, \316\224H=15.00kJ",
        .choice[8] = "q=18.68kJ, \316\224E=11.21kJ, \316\224H=0",
        .choice[9] = "q=18.68kJ, \316\224E=0, \316\224H=11.21kJ",
        .answer = "E",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "10567",
        .domain = "computer science",
        .title = "Computer Science challenge",
        .question = "Let F: K \\times R \\to MF:K\303\227R\342\206\222M be a secure PRF. For m \\in Mm\342\210"
        "\210M define E(k,m) = \\big[ r \\gets R,\\ \\text{output } \\big(r,\\ F(k,r) \\oplus m\\big)"
        "\\ \\big]E(k,m)=[r\342\206\220R, output (r, F(k,r)\342\212\225m) ] Is EE symantically secure"
        " under CPA?",
        .choice[0] = "It depends on the size of R",
        .choice[1] = "Yes, but only if R is large enough so r never repeats (w.h.p)",
        .choice[2] = "Yes, but only if M is small enough so m never repeats (w.h.p)",
        .choice[3] = "No, unless F is a weak PRF",
        .choice[4] = "No, unless F is a strong PRF",
        .choice[5] = "Yes, but only if K is large enough so k never repeats (w.h.p)",
        .choice[6] = "It depends on the size of K",
        .choice[7] = "No, there is always a CPA attack on this system",
        .choice[8] = "It depends on what F is used",
        .choice[9] = "Yes, whenever F is a secure PRF",
        .answer = "B",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "6955",
        .domain = "economics",
        .title = "Economics challenge",
        .question = "Which of the following is included in U.S. GDP? I.   The market value of movies made in Afri"
        "ca by U.S. citizens II.   The market value of olive oil made in Italy and sold in the United"
        " States. III.   The market value of blue jeans made in the United States and sold in Japan I"
        "V.   The market value of wine made in the United States by Canadian citizens",
        .choice[0] = "II and IV only",
        .choice[1] = "I, II and IV only",
        .choice[2] = "I only",
        .choice[3] = "IV only",
        .choice[4] = "I, II and III only",
        .choice[5] = "I and IV only",
        .choice[6] = "III and IV only",
        .choice[7] = "I and III only",
        .choice[8] = "II and III only",
        .choice[9] = "II III and IV only",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "11346",
        .domain = "engineering",
        .title = "Engineering challenge",
        .question = "Calculate the loss of head in a pipe (1/4) in. in diameter and 16 ft. long, when water flows"
        " at half the critical velocity. The critical velocity occurs when Reynolds number is 2500. T"
        "ake viscosity \\mu = 1.01 centipoise = 0.679 \303\227 10^-3 lbm/ft-sec.",
        .choice[0] = "3.12 in",
        .choice[1] = "7.5 in",
        .choice[2] = "0.8 in",
        .choice[3] = "6.3 in",
        .choice[4] = "2.5 in",
        .choice[5] = "5.0 in",
        .choice[6] = "1.75 in",
        .choice[7] = "4.2 in",
        .choice[8] = "4.8 in",
        .choice[9] = "2.0 in",
        .answer = "A",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "6500",
        .domain = "health",
        .title = "Health challenge",
        .question = "A 33-year-old man undergoes a radical thyroidectomy for thyroid cancer. During the operation"
        ", moderate hemorrhaging requires ligation of several vessels in the left side of the neck. P"
        "ostoperatively, serum studies show a calcium concentration of 7.5 mg/dL, albumin concentrati"
        "on of 4 g/dL, and parathyroid hormone concentration of 200 pg/mL. Damage to which of the fol"
        "lowing vessels caused the findings in this patient?",
        .choice[0] = "Branch of the costocervical trunk",
        .choice[1] = "Branch of the external carotid artery",
        .choice[2] = "Branch of the thyrocervical trunk",
        .choice[3] = "Tributary of the internal jugular vein",
        .answer = "C",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "4674",
        .domain = "history",
        .title = "History challenge",
        .question = "Recent discoveries of Homo naledi fossils in the Dinaledi Chamber of the Rising Star cave sy"
        "stem in South Africa suggest a mixture of post-cranial morphologies or mosaic of hominid tra"
        "its. One of the reasons this new species is controversial is that:",
        .choice[0] = "the Homo naledi fossils were found in the same layer as modern human fossils, suggesting the"
        "y lived at the same time.",
        .choice[1] = "the Homo naledi fossils have traits that do not fit into the current understanding of homini"
        "d evolution.",
        .choice[2] = "paleoanthropologists did not make the discovery and it was found in an inaccessible cave in "
        "South Africa.",
        .choice[3] = "the fossils were found in a region previously thought to be inhabited by a different species"
        " of Homo.",
        .choice[4] = "the Homo naledi fossils show traits of both Homo and Australopithecus, suggesting a potentia"
        "l hybrid species.",
        .choice[5] = "the methods used to extract the fossils from the cave were controversial and potentially dam"
        "aging to the specimens.",
        .choice[6] = "there is, as yet, no firm date that can be applied to the specimens.",
        .choice[7] = "the Homo naledi fossils have been shown to pre-date the earliest known hominid fossils by mi"
        "llions of years.",
        .choice[8] = "there are already two well-known species of Homo and it is unlikely there were more than two"
        " species at one time.",
        .choice[9] = "the researchers who discovered the fossils have been accused of falsifying their findings.",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "1467",
        .domain = "law",
        .title = "Law challenge",
        .question = "A motorcycle collector and his friend had known each other for years. The collector sent a l"
        "etter to the friend that said, \"I am willing to sell you my oldest motorcycle for $4,000.\""
        " The friend sent the collector an email saying, \"I accept your offer.\" The collector owned"
        " three motorcycles: a 1985 model, a 1987 model, and a 1992 model. The friend did not know ab"
        "out the 1985 model, which was the one the collector intended to sell. The collector was unaw"
        "are that the friend intended to buy the 1987 model. When the collector tendered the 1985 mod"
        "el, the friend refused to accept it. Is the friend likely to prevail in a breach of contract"
        " suit against the collector?",
        .choice[0] = "No, because it was the friend's responsibility to know the details about the collector's mot"
        "orcycles before agreeing to purchase.",
        .choice[1] = "No, because by not inquiring further about the meaning of the phrase \"oldest motorcycle,\" "
        "the friend assumed the risk of being mistaken.",
        .choice[2] = "No, the friend won't prevail as he misunderstood the term \"oldest motorcycle\".",
        .choice[3] = "Yes, so long as a reasonable person in the friend's position would have considered the lette"
        "r as referring to the 1987 model.",
        .choice[4] = "No, because neither the friend nor the collector had reason to know of the other's interpret"
        "ation of the words \"oldest motorcycle.\"",
        .choice[5] = "Yes, the friend will prevail as the collector falsely advertised the motorcycle model.",
        .choice[6] = "No, because the friend should have asked for clarification about which model was being sold "
        "before agreeing to the purchase.",
        .choice[7] = "Yes, because the friend could have reasonably assumed that the \"oldest motorcycle\" referre"
        "d to the 1987 model, given his knowledge.",
        .choice[8] = "Yes, because the friend had no reason to know of the collector's meaning and the collector c"
        "ould have avoided the confusion by more careful drafting.",
        .choice[9] = "Yes, because the collector should have clarified which model he was referring to in the lett"
        "er.",
        .answer = "E",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "8771",
        .domain = "math",
        .title = "Math challenge",
        .question = "7.4-11. Some dentists were interested in studying the fusion of embryonic rat palates by a s"
        "tandard transplantation technique. When no treatment is used, the probability of fusion equa"
        "ls approximately 0.89 . The dentists would like to estimate $p$, the probability of fusion, "
        "when vitamin A is lacking.\n(a) How large a sample $n$ of rat embryos is needed for $y / n "
        "\\pm 0.10$ to be a $95 \\%$ confidence interval for $p$ ?",
        .choice[0] = "70",
        .choice[1] = "50",
        .choice[2] = "45",
        .choice[3] = "40",
        .choice[4] = "30",
        .choice[5] = "55",
        .choice[6] = "$38$",
        .choice[7] = "65",
        .choice[8] = "60",
        .choice[9] = "25",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "5848",
        .domain = "other",
        .title = "Other challenge",
        .question = "Lake Trust a simple trust reported the following items of income and expense during the year"
        ": Dividend income $2500 Taxable interest income 2000 Capital gains (allocable to corpus) 500"
        "0 Accounting fees (allocable to income) (500) Trustee fees (allocable to income) (750) What "
        "is Lake's distributable net income?",
        .choice[0] = "$5,000",
        .choice[1] = "$6,000",
        .choice[2] = "$9,500",
        .choice[3] = "$2,000",
        .choice[4] = "$4,500",
        .choice[5] = "$10,000",
        .choice[6] = "$8,250",
        .choice[7] = "$7,500",
        .choice[8] = "$1,750",
        .choice[9] = "3250",
        .answer = "J",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "11120",
        .domain = "philosophy",
        .title = "Philosophy challenge",
        .question = " Use indirect truth tables to determine whether each set of propositions is consistent. If t"
        "he set is consistent, choose an option with a consistent valuation. (There may be other cons"
        "istent valuations.)\nD \342\211\241 E\nF \342\210\250 D\n~E \302\267 ~F",
        .choice[0] = "Consistent. Consistent valuation when D and E are true and F is true",
        .choice[1] = "Inconsistent",
        .choice[2] = "Consistent. Consistent valuation when D is true, E and F are false",
        .choice[3] = "Consistent. Consistent valuation when E is true, D and F are false",
        .choice[4] = "Consistent. Consistent valuation when D and E are true and F is false",
        .choice[5] = "Consistent. Consistent valuation when D, E are false and F is true",
        .choice[6] = "Consistent. Consistent valuation when D and F are true and E is false",
        .choice[7] = "Consistent. Consistent valuation when D, E, and F are true",
        .choice[8] = "Consistent. Consistent valuation when F is true, D and E are false",
        .choice[9] = "Consistent. Consistent valuation when D, E and F are false",
        .answer = "B",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "9487",
        .domain = "physics",
        .title = "Physics challenge",
        .question = "A gas sample is known to be a mixture of ethane and butane. A bulb having a $230.0 \\mathrm{"
        "~cm}^3$ capacity is filled with the gas to a pressure of $97.5 \\times 10^3 \\mathrm{~Pa}$ a"
        "t $23.1^{\\circ} \\mathrm{C}$. If the mass of the gas in the bulb is $0.3554 \\mathrm{~g}$, "
        "what is the mole percent of butane in the mixture?\n",
        .choice[0] = "70 %",
        .choice[1] = "27 %",
        .choice[2] = "50 %",
        .choice[3] = "10 %",
        .choice[4] = "60 %",
        .choice[5] = "15 %",
        .choice[6] = "25 %",
        .choice[7] = "32 %",
        .choice[8] = "45 %",
        .choice[9] = "38 %",
        .answer = "H",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "2398",
        .domain = "psychology",
        .title = "Psychology challenge",
        .question = "Susan is a first-grade student who is referred to the school psychologist because she will n"
        "ot remain seated at her desk. Her teacher reports that Susan is \"always getting up and walk"
        "ing around the room.\" What observation data is LEAST relevant to the referral concern?",
        .choice[0] = "Susan's academic performance",
        .choice[1] = "The teacher's teaching style",
        .choice[2] = "Number of students in the class",
        .choice[3] = "The school's disciplinary policies",
        .choice[4] = "Classroom environment",
        .choice[5] = "Peer comparison",
        .choice[6] = "Frequency",
        .choice[7] = "Intensity",
        .choice[8] = "Susan's diet",
        .choice[9] = "Duration",
        .answer = "H",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "MMLU-Pro",
        .id = "8863",
        .domain = "math",
        .title = "Math challenge",
        .question = "An investor deposits $1000 in an account paying interest at a rate of 8% compounded monthly,"
        " and also makes additional deposits of \\$25 per month. Find the balance in the account afte"
        "r 3 years.",
        .choice[0] = "$3200.67",
        .choice[1] = "$2050.89",
        .choice[2] = "$2100.45",
        .choice[3] = "$1950.32",
        .choice[4] = "$3000.59",
        .choice[5] = "$2500.00",
        .choice[6] = "2283.63 $",
        .choice[7] = "$2750.48",
        .choice[8] = "$2600.25",
        .choice[9] = "$1800.75",
        .answer = "G",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "OlympiadBench",
        .id = "1606",
        .domain = "Combinatorics",
        .title = "Olympiad problem 1606",
        .question = "Xenia and Sergey play the following game. Xenia thinks of a positive integer $N$ not exceedi"
        "ng 5000. Then she fixes 20 distinct positive integers $a_{1}, a_{2}, \\ldots, a_{20}$ such t"
        "hat, for each $k=1,2, \\ldots, 20$, the numbers $N$ and $a_{k}$ are congruent modulo $k$. By"
        " a move, Sergey tells Xenia a set $S$ of positive integers not exceeding 20 , and she tells "
        "him back the set $\\left\\{a_{k}: k \\in S\\right\\}$ without spelling out which number corr"
        "esponds to which index. How many moves does Sergey need to determine for sure the number Xen"
        "ia thought of?",
        .answer = "2",
        .answer_kind = EVAL_ANSWER_INTEGER,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1613",
        .domain = "Algebra",
        .title = "Olympiad problem 1613",
        .question = "Determine all positive integers $n$ satisfying the following condition: for every monic poly"
        "nomial $P$ of degree at most $n$ with integer coefficients, there exists a positive integer "
        "$k \\leq n$, and $k+1$ distinct integers $x_{1}, x_{2}, \\ldots, x_{k+1}$ such that\n\n\n\n$"
        "$\n\nP\\left(x_{1}\\right)+P\\left(x_{2}\\right)+\\cdots+P\\left(x_{k}\\right)=P\\left(x_{k+"
        "1}\\right) .\n\n$$\n\n\nNote. A polynomial is monic if the coefficient of the highest power "
        "is one.",
        .answer = "2",
        .answer_kind = EVAL_ANSWER_INTEGER,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1614",
        .domain = "Combinatorics",
        .title = "Olympiad problem 1614",
        .question = "Let $n$ be an integer greater than 1 and let $X$ be an $n$-element set. A non-empty collecti"
        "on of subsets $A_{1}, \\ldots, A_{k}$ of $X$ is tight if the union $A_{1} \\cup \\cdots \\cu"
        "p A_{k}$ is a proper subset of $X$ and no element of $X$ lies in exactly one of the $A_{i}$ "
        "s. Find the largest cardinality of a collection of proper non-empty subsets of $X$, no non-e"
        "mpty subcollection of which is tight.\n\n\n\nNote. A subset $A$ of $X$ is proper if $A \\neq"
        " X$. The sets in a collection are assumed to be distinct. The whole collection is assumed to"
        " be a subcollection.",
        .answer = "2n-2",
        .alias[0] = "2(n-1)",
        .answer_kind = EVAL_ANSWER_EXACT_TEXT,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1618",
        .domain = "Number Theory",
        .title = "Olympiad problem 1618",
        .question = "Determine all prime numbers $p$ and all positive integers $x$ and $y$ satisfying $x^{3}+y^{3"
        "}=$ $p(x y+p)$. Report each solution once as $(x,y,p)$ with $x \\leq y$.",
        .answer = "(1,8,19),(2,7,13),(4,5,7)",
        .alias[0] = "(4,5,7),(2,7,13),(1,8,19)",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1697",
        .domain = "Combinatorics",
        .title = "Olympiad problem 1697",
        .question = "On a square table of 2011 by 2011 cells we place a finite number of napkins that each cover "
        "a square of 52 by 52 cells. In each cell we write the number of napkins covering it, and we "
        "record the maximal number $k$ of cells that all contain the same nonzero number. Considering"
        " all possible napkin configurations, what is the largest value of $k$ ?",
        .answer = "3986729",
        .answer_kind = EVAL_ANSWER_INTEGER,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1709",
        .domain = "Number Theory",
        .title = "Olympiad problem 1709",
        .question = "For each positive integer $k$, let $t(k)$ be the largest odd divisor of $k$. Determine all p"
        "ositive integers $a$ for which there exists a positive integer $n$ such that all the differe"
        "nces\n\n$$\nt(n+a)-t(n), \\quad t(n+a+1)-t(n+1), \\quad \\ldots, \\quad t(n+2 a-1)-t(n+a-1)"
        "\n$$\n\nare divisible by 4 .",
        .answer = "1,3,5",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1716",
        .domain = "Algebra",
        .title = "Olympiad problem 1716",
        .question = "Let $x_{1}, \\ldots, x_{100}$ be nonnegative real numbers such that $x_{i}+x_{i+1}+x_{i+2} "
        "\\leq 1$ for all $i=1, \\ldots, 100$ (we put $x_{101}=x_{1}, x_{102}=x_{2}$ ). Find the maxi"
        "mal possible value of the sum\n\n$$\nS=\\sum_{i=1}^{100} x_{i} x_{i+2}\n$$",
        .answer = "25/2",
        .alias[0] = "12.5",
        .answer_kind = EVAL_ANSWER_RATIONAL,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1736",
        .domain = "Number Theory",
        .title = "Olympiad problem 1736",
        .question = "Find the least positive integer $n$ for which there exists a set $\\left\\{s_{1}, s_{2}, \\l"
        "dots, s_{n}\\right\\}$ consisting of $n$ distinct positive integers such that\n\n$$\n\\left("
        "1-\\frac{1}{s_{1}}\\right)\\left(1-\\frac{1}{s_{2}}\\right) \\ldots\\left(1-\\frac{1}{s_{n}}"
        "\\right)=\\frac{51}{2010}\n$$",
        .answer = "39",
        .answer_kind = EVAL_ANSWER_INTEGER,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1750",
        .domain = "Combinatorics",
        .title = "Olympiad problem 1750",
        .question = "A diagonal of a regular 2006-gon is called odd if its endpoints divide the boundary into two"
        " parts, each composed of an odd number of sides. Sides are also regarded as odd diagonals.\n"
        "\nSuppose the 2006-gon has been dissected into triangles by 2003 nonintersecting diagonals. "
        "Find the maximum possible number of isosceles triangles with two odd sides.",
        .answer = "1003",
        .answer_kind = EVAL_ANSWER_INTEGER,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "OlympiadBench",
        .id = "1766",
        .domain = "Number Theory",
        .title = "Olympiad problem 1766",
        .question = "Determine all pairs $(x, y)$ of integers satisfying the equation\n\n$$\n1+2^{x}+2^{2 x+1}=y^"
        "{2}\n$$",
        .answer = "(0,2),(0,-2),(4,23),(4,-23)",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 16000,
    },
    {
        .source = "LiveBench",
        .id = "c833574b0ff6e8609bee19b36fabc48501cdcd9a64b99cd0422f54ada1acc53d",
        .domain = "zebra_puzzle",
        .title = "Zebra logic puzzle",
        .question = "There are 4 people standing in a line. From left to right, they are numbered 1 to 4.\nEach p"
        "erson has a set of attributes: Beverage, Job, Movie-Genre, Transport.\nThe attributes have t"
        "he following possible values:\nBeverage: lemonade, fanta, iced-tea, hot-chocolate\nJob: soci"
        "al-worker, software-developer, coach, accountant\nMovie-Genre: martial-arts, horror, documen"
        "tary, satire\nTransport: skateboard, train, ship, helicopter\nEach person has a unique value"
        " for each attribute.\nYou know the following about the people:\nThe person who watches docum"
        "entary is somewhere between the person who watches martial-arts and the person who is a acco"
        "untant\nThe person who travels by skateboard is not anywhere to the left of the person who t"
        "ravels by train\nThe person who watches martial-arts and the person who drinks iced-tea have"
        " the same parity positions\nThe person who watches satire is not anywhere to the right of th"
        "e person who watches horror\nThe person who is a accountant and the person who drinks hot-ch"
        "ocolate have the same parity positions\nThe person who travels by train and the person who d"
        "rinks hot-chocolate have different parity positions\nEither the person who is a coach is the"
        " same as the person who travels by helicopter or the person who is a coach is the same as th"
        "e person who watches documentary, but not both\nThe person who watches satire and the person"
        " who is a accountant have different parity positions\nThe person who watches satire is not a"
        "nywhere to the right of the person who watches documentary\nThe person who travels by skateb"
        "oard is the same as the person who watches horror or the person who drinks fanta is the same"
        " as the person who travels by skateboard or both\nThe person who drinks fanta is somewhere b"
        "etween the person who watches satire and the person who watches martial-arts\nThe person who"
        " is a social-worker and the person who travels by helicopter have the same parity positions"
        "\nThe person who is a accountant is not anywhere to the left of the person who travels by sh"
        "ip\nThe person who drinks iced-tea is not anywhere to the right of the person who drinks fan"
        "ta\nIn the above, 'parity position' means the evenness or oddness of the person's position i"
        "n the line.\nGiven this information, answer the following questions:\nWhat is the movie genr"
        "e of the person who is a software-developer?\nWhat movie genre does the person who drinks ho"
        "t-chocolate watch?\nWhat movie genre does the person in position 2 watch?\nAt what position "
        "is the person who watches horror?",
        .answer = "satire, martial-arts, horror, 2",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 8192,
    },
    {
        .source = "LiveBench",
        .id = "b7b7e331df4614775210ef5e524ed7f20c5fec6655fb0dd179c94e4d71170862",
        .domain = "zebra_puzzle",
        .title = "Zebra logic puzzle",
        .question = "There are 3 people standing in a line. From left to right, they are numbered 1 to 3.\nEach p"
        "erson has a set of attributes: Beverage, Food, Movie-Genre, Nationality.\nThe attributes hav"
        "e the following possible values:\nBeverage: mirinda, cola, water\nFood: cauliflower, radish,"
        " zucchini\nMovie-Genre: family, spy, crime\nNationality: brazilian, german, canadian\nEach p"
        "erson has a unique value for each attribute.\nYou know the following about the people:\nThe "
        "person who is brazilian is somewhere between the person who is german and the person who dri"
        "nks cola\nThe person who eats cauliflower is not anywhere to the right of the person who dri"
        "nks cola\nThe person who drinks mirinda is somewhere between the person who drinks cola and "
        "the person who is german\nThe person who watches family and the person who eats radish have "
        "the same parity positions\nThe person who watches spy is not anywhere to the right of the pe"
        "rson who watches family\nEither the person who is canadian is the same as the person who wat"
        "ches spy or the person who eats cauliflower is the same as the person who is canadian, but n"
        "ot both\nThe person who eats cauliflower is not anywhere to the right of the person who eats"
        " radish\nIn the above, 'parity position' means the evenness or oddness of the person's posit"
        "ion in the line.\nGiven this information, answer the following questions:\nWhat movie genre "
        "does the person in position 1 watch?\nAt what position is the person who watches crime?\nAt "
        "what position is the person who eats zucchini?\nWhat movie genre does the person in position"
        " 3 watch?",
        .answer = "crime, 1, 2, family",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 8192,
    },
    {
        .source = "LiveBench",
        .id = "0daa7ca38beec4441b9d5c04d0b98912322926f0a3ac28a5097889d4ed83506f",
        .domain = "web_of_lies_v2",
        .title = "Web of lies",
        .question = "In this question, assume each person either always tells the truth or always lies. Tala is a"
        "t the movie theater. The person at the restaurant says the person at the aquarium lies. Ayaa"
        "n is at the aquarium. Ryan is at the botanical garden. The person at the park says the perso"
        "n at the art gallery lies. The person at the museum tells the truth. Zara is at the museum. "
        "Jake is at the art gallery. The person at the art gallery says the person at the theater lie"
        "s. Beatriz is at the park. The person at the movie theater says the person at the train stat"
        "ion lies. Nadia is at the campground. The person at the campground says the person at the ar"
        "t gallery tells the truth. The person at the theater lies. The person at the amusement park "
        "says the person at the aquarium tells the truth. Grace is at the restaurant. The person at t"
        "he aquarium thinks their friend is lying. Nia is at the theater. Kehinde is at the train sta"
        "tion. The person at the theater thinks their friend is lying. The person at the botanical ga"
        "rden says the person at the train station tells the truth. The person at the aquarium says t"
        "he person at the campground tells the truth. The person at the aquarium saw a firetruck. The"
        " person at the train station says the person at the amusement park lies. Mateo is at the amu"
        "sement park. Does the person at the train station tell the truth? Does the person at the amu"
        "sement park tell the truth? Does the person at the aquarium tell the truth?",
        .answer = "no, yes, yes",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 8192,
    },
    {
        .source = "LiveBench",
        .id = "9ee37b9a04ab050936c86b2c5bb7abbaa0bc0e737d59a7bff9ba11e9b4069c1d",
        .domain = "web_of_lies_v2",
        .title = "Web of lies",
        .question = "In this question, assume each person either always tells the truth or always lies. Liam is a"
        "t the movie theater. The person at the beach says the person at the restaurant tells the tru"
        "th. Kehinde is at the aquarium. The person at the amusement park saw a firetruck. The person"
        " at the aquarium tells the truth. Luna is at the library. Jaxon is at the amusement park. Th"
        "e person at the amusement park says the person at the beach tells the truth. Anika is at the"
        " restaurant. The person at the barbershop tells the truth. The person at the observatory say"
        "s the person at the planetarium lies. Hiroshi is at the beach. Isabella is at the planetariu"
        "m. Nia is at the barbershop. The person at the movie theater says the person at the observat"
        "ory lies. The person at the restaurant tells the truth. Max is at the observatory. The perso"
        "n at the library says the person at the amusement park tells the truth. The person at the pl"
        "anetarium says the person at the library tells the truth. Does the person at the movie theat"
        "er tell the truth? Does the person at the observatory tell the truth? Does the person at the"
        " planetarium tell the truth?",
        .answer = "yes, no, yes",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 8192,
    },
    {
        .source = "LiveBench",
        .id = "d7071c9ff5d9779e7ab955366d0ae8db40f785aadfe7ff0b5a7ede98c05c44ea",
        .domain = "web_of_lies_v2",
        .title = "Web of lies",
        .question = "In this question, assume each person either always tells the truth or always lies. Liam is a"
        "t the restaurant. The person at the observatory says the person at the botanical garden lies"
        ". Elowen is at the vineyard. The person at the library says the person at the botanical gard"
        "en tells the truth. The person at the train station tells the truth. The person at the botan"
        "ical garden saw a firetruck. The person at the botanical garden says the person at the train"
        " station lies. Jake is at the aquarium. Soren is at the farm. Theo is at the gym. The person"
        " at the train station saw a firetruck. Devika is at the train station. Kehinde is at the lib"
        "rary. The person at the restaurant lies. The person at the farm says the person at the train"
        " station lies. The person at the gym says the person at the train station lies. Hiroshi is a"
        "t the botanical garden. The person at the vineyard lies. The person at the aquarium says the"
        " person at the library tells the truth. Quan is at the observatory. Does the person at the l"
        "ibrary tell the truth? Does the person at the botanical garden tell the truth? Does the pers"
        "on at the train station tell the truth?",
        .answer = "no, no, yes",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 8192,
    },
    {
        .source = "NIST Juliet",
        .id = "juliet-cwe190-int-multiply-01",
        .domain = "CWE-190",
        .title = "Integer multiplication overflow",
        .question = "The following C function is intended to allocate space for count integers. Identify the line"
        " containing the primary weakness.\n\n 1  int *make_array(size_t count) {\n 2      int *resul"
        "t = NULL;\n 3      if (count == 0)\n 4          return NULL;\n 5      size_t bytes;\n 6     "
        " if (count > SIZE_MAX / sizeof(int))\n 7          return NULL;\n 8      bytes = count * size"
        "of(int);\n 9      result = malloc(bytes);\n10      if (result == NULL)\n11          return N"
        "ULL;\n12      memset(result, 0, count * count * sizeof(int));\n13      return result;\n14  }",
        .answer = "12",
        .answer_kind = EVAL_ANSWER_LINE_SET,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "NIST Juliet",
        .id = "juliet-cwe121-array-index-01",
        .domain = "CWE-121",
        .title = "Stack array index validation",
        .question = "The following C function should update one element of a fixed-size stack array. Identify the"
        " smallest exact set of line numbers containing the primary weakness.\n\n 1  int update_value"
        "(int index, int value) {\n 2      int data[10] = {0};\n 3      if (index < 10)\n 4          "
        "data[index] = value;\n 5      return data[0];\n 6  }",
        .answer = "3-4",
        .answer_kind = EVAL_ANSWER_LINE_SET,
        .suites = EVAL_SUITE_HARD | EVAL_SUITE_HARD_SMOKE,
        .max_tokens = 4096,
    },
    {
        .source = "NIST Juliet",
        .id = "juliet-cwe401-realloc-01",
        .domain = "CWE-401",
        .title = "Lost allocation after realloc",
        .question = "The following C function grows an allocated buffer. Identify the line containing the primary"
        " weakness.\n\n 1  char *grow(char *buf, size_t size) {\n 2      if (size > SIZE_MAX / 2)\n 3"
        "          return NULL;\n 4      size *= 2;\n 5      buf = realloc(buf, size);\n 6      if (b"
        "uf == NULL)\n 7          return NULL;\n 8      return buf;\n 9  }",
        .answer = "5",
        .answer_kind = EVAL_ANSWER_LINE_SET,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "NIST Juliet",
        .id = "juliet-cwe476-null-after-deref-01",
        .domain = "CWE-476",
        .title = "Null check after dereference",
        .question = "The following C function reads the first byte of a caller-supplied buffer. Identify the line"
        " containing the primary weakness.\n\n 1  int first_byte(const unsigned char *p) {\n 2      i"
        "nt value = p[0];\n 3      if (p == NULL)\n 4          return -1;\n 5      return value;\n 6 "
        " }",
        .answer = "2",
        .answer_kind = EVAL_ANSWER_LINE_SET,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "NIST Juliet",
        .id = "juliet-cwe369-int-divide-01",
        .domain = "CWE-369",
        .title = "Unchecked integer division",
        .question = "The following C function computes a ratio from caller-supplied integers. Identify the line c"
        "ontaining the primary weakness.\n\n 1  int ratio(int total, int parts) {\n 2      if (total "
        "< 0)\n 3          return -1;\n 4      return total / parts;\n 5  }",
        .answer = "4",
        .answer_kind = EVAL_ANSWER_LINE_SET,
        .suites = EVAL_SUITE_HARD,
        .max_tokens = 4096,
    },
    {
        .source = "LifeOS",
        .id = "lifeos-finance-card-bill-01",
        .domain = "finance",
        .title = "Credit card bill with installments",
        .question = "Dana Okafor's assistant tracks one credit card with these rules:\n- The statement closes o"
        "n the 5th of each month and the bill is due on the 12th of the same month.\n- A purchase m"
        "ade on or before the closing day goes on that month's bill; a purchase made after the clos"
        "ing day goes on the next month's bill.\n- An n-installment purchase is split evenly in eur"
        "o cents; any leftover cents go to the first installment. Installment k lands on the bill k"
        "-1 months after the first one.\n- A refund is credited in full on the first bill whose sta"
        "tement closes on or after the refund date.\n- A monthly card fee of 4.50 is charged on eve"
        "ry bill.\n\nCard activity:\n- 2026-10-03: laptop, 1234.57, 6 installments\n- 2026-11-05: h"
        "eadphones, 74.30, 1 installment\n- 2026-11-06: sofa, 999.99, 3 installments\n- 2026-12-20:"
        " gift, 136.00, 2 installments\n- 2026-12-28: refund for a returned jacket, 40.00\n\nWhat i"
        "s the total of the bill due on 2027-01-12? Give a plain number in euros with two decimals "
        "and no currency symbol.",
        .answer = "571.59",
        .alias[0] = "\342\202\254571.59",
        .alias[1] = "571.59eur",
        .alias[2] = "eur571.59",
        .answer_kind = EVAL_ANSWER_EXACT_TEXT,
        .suites = EVAL_SUITE_LIFEOS,
        .max_tokens = 16384,
    },
    {
        .source = "LifeOS",
        .id = "lifeos-cron-dst-01",
        .domain = "dates",
        .title = "UTC cron seen from Lisbon across DST",
        .question = "Dana lives in Lisbon (time zone Europe/Lisbon) and uses an assistant whose scheduler runs "
        "in UTC. Her weekday reminder is the cron entry `30 6 * * 1-5`, interpreted in UTC.\n\nAt w"
        "hat Lisbon local wall-clock times does the reminder fire on these three days, in this orde"
        "r:\n1. Friday 2027-03-26\n2. Monday 2027-03-29\n3. Monday 2027-11-01\n\nWrite each time as"
        " HH:MM in 24-hour format, separated by commas.",
        .answer = "06:30, 07:30, 06:30",
        .alias[0] = "6:30, 7:30, 6:30",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_LIFEOS,
        .max_tokens = 16384,
    },
    {
        .source = "LifeOS",
        .id = "lifeos-gym-pr-progression-01",
        .domain = "gym",
        .title = "Bench press PR and next load",
        .question = "Dana's gym assistant applies these rules to the barbell bench press:\n- Estimated one-rep "
        "max (e1RM) uses the Epley formula: weight * (1 + reps / 30).\n- Only working sets count to"
        "ward personal records. Dropset sets and spotter-assisted sets never count.\n- The current "
        "PR date is the date the current best e1RM was first reached; tying it later does not move "
        "the PR date.\n- Progression for the next session uses the last session's three working set"
        "s at the top weight with a 6-8 rep target: if all three reached 8 reps, add 2.5 kg; if any"
        " of them fell below 6 reps, reduce the weight by 5% and round to the nearest 1.25 kg; othe"
        "rwise keep the weight.\n\nLog (weight kg x reps):\n- 2026-09-23: 65x8, 65x8, 65x8\n- 2026-"
        "09-30: 67.5x8, 67.5x8, 67.5x8, dropset 50x12\n- 2026-10-07: 70x6, 70x6, 70x7\n- 2026-10-14"
        ": 70x8, 70x7, 70x6, spotter-assisted 80x4\n- 2026-10-21: 70x8, 70x8, 70x8, dropset 60x15\n"
        "\nGive the current PR date (YYYY-MM-DD) and the weight in kg for the next session, separat"
        "ed by a comma.",
        .answer = "2026-10-14, 72.5",
        .alias[0] = "2026-10-14, 72.50",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_LIFEOS,
        .max_tokens = 16384,
    },
    {
        .source = "LifeOS",
        .id = "lifeos-notes-conflict-01",
        .domain = "grounding",
        .title = "Conflicting notes about leg day",
        .question = "Below are Dana's notes, oldest first. Notes marked \"draft\" were never approved and do no"
        "t change the plan.\n\n- 2026-03-02: Leg day is Tuesday.\n- 2026-07-15 (draft): Thinking ab"
        "out moving leg day to Wednesday.\n- 2026-10-01: Knee check booked for 2026-10-20.\n- 2026-"
        "10-03: Physio: move leg day to Thursday, starting the week after the knee check.\n- 2026-1"
        "0-10: Knee check moved earlier to 2026-10-13. It went fine.\n- 2026-10-17: The gym is clos"
        "ed on Thursday 2026-10-22 for maintenance. In any week where the gym is closed on leg day,"
        " train legs the day after instead.\n\nToday is Monday 2026-10-19. On which day this week s"
        "hould Dana train legs?",
        .choice[0] = "Monday",
        .choice[1] = "Tuesday",
        .choice[2] = "Wednesday",
        .choice[3] = "Thursday",
        .choice[4] = "Friday",
        .choice[5] = "Saturday",
        .answer = "E",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_LIFEOS,
        .max_tokens = 16384,
    },
    {
        .source = "LifeOS",
        .id = "lifeos-notes-missing-fact-01",
        .domain = "grounding",
        .title = "Birthdays that the notes do and do not state",
        .question = "Below are Dana's notes, oldest first. Use only these notes; do not guess.\n\n- 2026-02-11:"
        " Mum's birthday is 3 September. Book the restaurant early.\n- 2026-04-02: Lena's birthday "
        "is 14 May.\n- 2026-05-20: Correction: I had Lena's birthday wrong, it is 15 May. She laugh"
        "ed about it.\n- 2026-06-18: Bought Tomas a book to bring to Lena's housewarming party on 2"
        "0 June.\n- 2026-06-21: Tomas loved the book. He turns into a different person when he talk"
        "s about birds.\n\nGive, in this order, the birthday of Lena, of Dana's mum, and of Tomas a"
        "s MM-DD, separated by commas. For any birthday the notes do not state, write NOT_IN_NOTES "
        "instead.",
        .answer = "05-15, 09-03, NOT_IN_NOTES",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_LIFEOS,
        .max_tokens = 16384,
    },
    {
        .source = "LifeOS",
        .id = "lifeos-tool-choice-01",
        .domain = "tools",
        .title = "Pick tools and arguments from a request",
        .question = "An assistant has these tools:\n- set_timer(minutes, label): starts a focus countdown that "
        "blocks notifications until it ends.\n- schedule_reminder(when, text): sends one reminder a"
        "t a local date and time, `when` written as YYYY-MM-DDTHH:MM.\n- create_cron(expression, te"
        "xt): creates a recurring reminder.\n- log_expense(amount, category): records money already"
        " spent; amount is a plain number with two decimals.\n- save_note(text): saves a note for l"
        "ater review.\n\nThe current local time is Saturday 2026-11-21 23:40. The user writes:\n\"R"
        "emind me in 45 minutes to text Lena back, no focus timer please, I'm out. Oh and the taxi "
        "here was 12.4, log it.\"\n\nList the tool calls the assistant should make, in the order th"
        "e user asked for them. For each call give the tool name followed by only its first argumen"
        "t (`when` for a reminder, `amount` for an expense); omit every other argument. Separate ev"
        "ery item with a comma, for example: tool_a, arg, tool_b, arg",
        .answer = "schedule_reminder, 2026-11-22T00:25, log_expense, 12.40",
        .alias[0] = "schedule_reminder, 2026-11-22T00:25, log_expense, 12.4",
        .answer_kind = EVAL_ANSWER_ORDERED_SEQUENCE,
        .suites = EVAL_SUITE_LIFEOS,
        .max_tokens = 16384,
    },
    {
        .source = "MMLU-ProX",
        .id = "en-8494",
        .domain = "en",
        .title = "Math (en)",
        .question = "The amount of rainfall per month in a certain city is approximately normally distributed, "
        "with a mean of six inches and a standard deviation of 1.6 inches. Which of the following i"
        "s the highest amount of rainfall, in inches, this city could have this month for the month"
        " to be among the 10percent driest months the city has seen?",
        .choice[0] = "3.5",
        .choice[1] = "7.6",
        .choice[2] = "8.05",
        .choice[3] = "3.37",
        .choice[4] = "2.3",
        .choice[5] = "4.6",
        .choice[6] = "5.2",
        .choice[7] = "3.95",
        .choice[8] = "6.1",
        .choice[9] = "2.8",
        .answer = "H",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "pt-8494",
        .domain = "pt",
        .title = "Math (pt)",
        .question = "A quantidade de chuva por m\303\252s em uma determinada cidade tem distribui\303\247\303\243"
        "o aproximadamente normal, com m\303\251dia de seis polegadas e desvio padr\303\243o de 1,6"
        " polegadas. Qual das seguintes alternativas representa a maior quantidade de chuva, em pol"
        "egadas, que esta cidade poderia ter neste m\303\252s para estar entre os 10% dos meses mai"
        "s secos que a cidade j\303\241 viu?",
        .choice[0] = "3,5",
        .choice[1] = "7,6",
        .choice[2] = "8,05",
        .choice[3] = "3,37",
        .choice[4] = "2,3",
        .choice[5] = "4,6",
        .choice[6] = "5,2",
        .choice[7] = "3,95",
        .choice[8] = "6,1",
        .choice[9] = "2,8",
        .answer = "H",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "zh-8494",
        .domain = "zh",
        .title = "Math (zh)",
        .question = "\346\237\220\344\270\252\345\237\216\345\270\202\347\232\204\346\234\210\351\231\215\351\233"
        "\250\351\207\217\350\277\221\344\274\274\345\221\210\346\255\243\346\200\201\345\210\206\345"
        "\270\203\357\274\214\345\271\263\345\235\207\345\200\274\344\270\2726\350\213\261\345\257\270"
        "\357\274\214\346\240\207\345\207\206\345\267\256\344\270\2721.6\350\213\261\345\257\270\343"
        "\200\202\344\273\245\344\270\213\345\223\252\344\270\252\346\230\257\350\257\245\345\237\216"
        "\345\270\202\346\234\254\346\234\210\345\217\257\350\203\275\345\207\272\347\216\260\347\232"
        "\204\346\234\200\351\253\230\351\231\215\351\233\250\351\207\217\357\274\210\344\273\245\350"
        "\213\261\345\257\270\350\256\241\357\274\211\357\274\214\344\275\277\345\276\227\350\277\231"
        "\344\270\252\346\234\210\346\210\220\344\270\272\350\257\245\345\237\216\345\270\202\347\273"
        "\217\345\216\206\350\277\207\347\232\204\346\234\200\345\271\262\347\207\245\347\232\20410"
        "%\347\232\204\346\234\210\344\273\275\344\271\213\344\270\200\357\274\237",
        .choice[0] = "3.5",
        .choice[1] = "7.6",
        .choice[2] = "8.05",
        .choice[3] = "3.37",
        .choice[4] = "2.3",
        .choice[5] = "4.6",
        .choice[6] = "5.2",
        .choice[7] = "3.95",
        .choice[8] = "6.1",
        .choice[9] = "2.8",
        .answer = "H",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "sw-8494",
        .domain = "sw",
        .title = "Math (sw)",
        .question = "Kipi kati ya vifuatavyo ni kiwango cha juu zaidi cha mvua, kwa inchi, ambacho jiji hili li"
        "naweza kuwa nacho mwezi huu ili mwezi huo uwe miongoni mwa miezi 10 ya asilimia ya ukame z"
        "aidi ambayo jiji limewahi kuona?",
        .choice[0] = "3.5",
        .choice[1] = "7.6",
        .choice[2] = "8.05",
        .choice[3] = "3.37",
        .choice[4] = "2.3",
        .choice[5] = "4.6",
        .choice[6] = "5.2",
        .choice[7] = "3.95",
        .choice[8] = "6.1",
        .choice[9] = "2.8",
        .answer = "H",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "en-9212",
        .domain = "en",
        .title = "Physics (en)",
        .question = "A rocket has a length L = 600 m measured at rest on the earth. It moves directly away from"
        " the earth at constant velocity, and a pulse of radar is sent from earth and reflected bac"
        "k by devices at the nose and tail of the rocket. The reflected signal from the tail is det"
        "ected on earth 200 sec after emission and the signal from the nose 17.4 \303\227 10^-6 sec"
        " later. Calculate the distance of the rocket from the earth and the velocity of the rocket"
        " with respect to the earth.",
        .choice[0] = "R = 3.2 \303\227 10^10 m, v = 2.40 \303\227 10^8 m/sec",
        .choice[1] = "R = 2 \303\227 10^10 m, v = 1.63 \303\227 10^8 m/sec",
        .choice[2] = "R = 3.5 \303\227 10^10 m, v = 2.50 \303\227 10^8 m/sec",
        .choice[3] = "R = 4.5 \303\227 10^10 m, v = 3.00 \303\227 10^8 m/sec",
        .choice[4] = "R = 2.5 \303\227 10^10 m, v = 2.00 \303\227 10^8 m/sec",
        .choice[5] = "R = 1.5 \303\227 10^10 m, v = 1.50 \303\227 10^8 m/sec",
        .choice[6] = "R = 4 \303\227 10^10 m, v = 2.63 \303\227 10^8 m/sec",
        .choice[7] = "R = 1 \303\227 10^10 m, v = 1.13 \303\227 10^8 m/sec",
        .choice[8] = "R = 3 \303\227 10^10 m, v = 2.13 \303\227 10^8 m/sec",
        .choice[9] = "R = 2.8 \303\227 10^10 m, v = 1.90 \303\227 10^8 m/sec",
        .answer = "I",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "pt-9212",
        .domain = "pt",
        .title = "Physics (pt)",
        .question = "Um foguete tem um comprimento L = 600 m medido em repouso na Terra. Ele se move diretament"
        "e para longe da Terra com velocidade constante, e um pulso de radar \303\251 enviado da Te"
        "rra e refletido de volta por dispositivos no nariz e na cauda do foguete. O sinal refletid"
        "o da cauda \303\251 detectado na Terra 200 seg ap\303\263s a emiss\303\243o e o sinal do n"
        "ariz 17,4 \303\227 10^-6 seg depois. Calcule a dist\303\242ncia do foguete at\303\251 a Te"
        "rra e a velocidade do foguete em rela\303\247\303\243o \303\240 Terra.",
        .choice[0] = "R = 3,2 \303\227 10^10 m, v = 2,40 \303\227 10^8 m/seg",
        .choice[1] = "R = 2 \303\227 10^10 m, v = 1,63 \303\227 10^8 m/seg",
        .choice[2] = "R = 3,5 \303\227 10^10 m, v = 2,50 \303\227 10^8 m/seg",
        .choice[3] = "R = 4,5 \303\227 10^10 m, v = 3,00 \303\227 10^8 m/seg",
        .choice[4] = "R = 2,5 \303\227 10^10 m, v = 2,00 \303\227 10^8 m/seg",
        .choice[5] = "R = 1,5 \303\227 10^10 m, v = 1,50 \303\227 10^8 m/seg",
        .choice[6] = "R = 4 \303\227 10^10 m, v = 2,63 \303\227 10^8 m/seg",
        .choice[7] = "R = 1 \303\227 10^10 m, v = 1,13 \303\227 10^8 m/seg",
        .choice[8] = "R = 3 \303\227 10^10 m, v = 2,13 \303\227 10^8 m/seg",
        .choice[9] = "R = 2,8 \303\227 10^10 m, v = 1,90 \303\227 10^8 m/seg",
        .answer = "I",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "zh-9212",
        .domain = "zh",
        .title = "Physics (zh)",
        .question = "\344\270\200\346\236\232\347\201\253\347\256\255\345\234\250\345\234\260\347\220\203\344\270"
        "\212\351\235\231\346\255\242\346\265\213\351\207\217\346\227\266\347\232\204\351\225\277\345"
        "\272\246L\344\270\272600\347\261\263\343\200\202\345\256\203\344\273\245\346\201\222\345\256"
        "\232\351\200\237\345\272\246\347\233\264\346\216\245\350\277\234\347\246\273\345\234\260\347"
        "\220\203\357\274\214\344\273\216\345\234\260\347\220\203\345\217\221\345\207\272\344\270\200"
        "\344\270\252\351\233\267\350\276\276\350\204\211\345\206\262\357\274\214\347\224\261\347\201"
        "\253\347\256\255\345\244\264\351\203\250\345\222\214\345\260\276\351\203\250\347\232\204\350"
        "\243\205\347\275\256\345\217\215\345\260\204\345\233\236\346\235\245\343\200\202\345\260\276"
        "\351\203\250\345\217\215\345\260\204\347\232\204\344\277\241\345\217\267\345\234\250\345\217"
        "\221\345\260\204\345\220\216200\347\247\222\350\242\253\345\234\260\347\220\203\346\216\242"
        "\346\265\213\345\210\260\357\274\214\350\200\214\345\244\264\351\203\250\347\232\204\344\277"
        "\241\345\217\267\345\234\25017.4\303\22710^-6\347\247\222\344\271\213\345\220\216\350\242\253"
        "\346\216\242\346\265\213\345\210\260\343\200\202\350\256\241\347\256\227\347\201\253\347\256"
        "\255\350\267\235\347\246\273\345\234\260\347\220\203\347\232\204\350\267\235\347\246\273\345"
        "\222\214\347\201\253\347\256\255\347\233\270\345\257\271\344\272\216\345\234\260\347\220\203"
        "\347\232\204\351\200\237\345\272\246\343\200\202",
        .choice[0] = "R = 3.2\303\22710^10\347\261\263\357\274\214v = 2.40\303\22710^8\347\261\263/\347\247\222",
        .choice[1] = "R = 2\303\22710^10\347\261\263\357\274\214v = 1.63\303\22710^8\347\261\263/\347\247\222",
        .choice[2] = "R = 3.5\303\22710^10\347\261\263\357\274\214v = 2.50\303\22710^8\347\261\263/\347\247\222",
        .choice[3] = "R = 4.5\303\22710^10\347\261\263\357\274\214v = 3.00\303\22710^8\347\261\263/\347\247\222",
        .choice[4] = "R = 2.5\303\22710^10\347\261\263\357\274\214v = 2.00\303\22710^8\347\261\263/\347\247\222",
        .choice[5] = "R = 1.5\303\22710^10\347\261\263\357\274\214v = 1.50\303\22710^8\347\261\263/\347\247\222",
        .choice[6] = "R = 4\303\22710^10\347\261\263\357\274\214v = 2.63\303\22710^8\347\261\263/\347\247\222",
        .choice[7] = "R = 1\303\22710^10\347\261\263\357\274\214v = 1.13\303\22710^8\347\261\263/\347\247\222",
        .choice[8] = "R = 3\303\22710^10\347\261\263\357\274\214v = 2.13\303\22710^8\347\261\263/\347\247\222",
        .choice[9] = "R = 2.8\303\22710^10\347\261\263\357\274\214v = 1.90\303\22710^8\347\261\263/\347\247\222",
        .answer = "I",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "sw-9212",
        .domain = "sw",
        .title = "Physics (sw)",
        .question = "Roketi ina urefu L = 600 m iliyopimwa ikiwa imetulia kwenye Dunia. Inasogea moja kwa moja "
        "mbali na Dunia kwa kasi ya kudumu, na mpigo wa rada unatumwa kutoka Dunia na kurejeshwa na"
        " vifaa vilivyoko kwenye pua na mkia wa roketi. Ishara iliyorejeshwa kutoka kwa mkia hugund"
        "uliwa kwenye Dunia sekunde 200 baada ya kutumwa na ishara kutoka kwa pua sekunde 17.4 \303"
        "\227 10^-6 baadaye. Hesabu umbali wa roketi kutoka Dunia na kasi ya roketi ikilinganishwa "
        "na Dunia.",
        .choice[0] = "R = 3.2 \303\227 10^10 m, v = 2.40 \303\227 10^8 m/sec",
        .choice[1] = "R = 2 \303\227 10^10 m, v = 1.63 \303\227 10^8 m/sec",
        .choice[2] = "R = 3.5 \303\227 10^10 m, v = 2.50 \303\227 10^8 m/sec",
        .choice[3] = "R = 4.5 \303\227 10^10 m, v = 3.00 \303\227 10^8 m/sec",
        .choice[4] = "R = 2.5 \303\227 10^10 m, v = 2.00 \303\227 10^8 m/sec",
        .choice[5] = "R = 1.5 \303\227 10^10 m, v = 1.50 \303\227 10^8 m/sec",
        .choice[6] = "R = 4 \303\227 10^10 m, v = 2.63 \303\227 10^8 m/sec",
        .choice[7] = "R = 1 \303\227 10^10 m, v = 1.13 \303\227 10^8 m/sec",
        .choice[8] = "R = 3 \303\227 10^10 m, v = 2.13 \303\227 10^8 m/sec",
        .choice[9] = "R = 2.8 \303\227 10^10 m, v = 1.90 \303\227 10^8 m/sec",
        .answer = "I",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "en-860",
        .domain = "en",
        .title = "Law (en)",
        .question = "A federal grand jury was investigating a corporation whose tanker ship had spilled crude o"
        "il into environmentally sensitive waters. The grand jury issued a subpoena requiring the c"
        "orporation to produce all emails and internal documents regarding the corporation's knowle"
        "dge of the risks of an oil spill. The corporation has objected, citing its Fifth Amendment"
        " privilege against self-incrimination. Can the subpoena be enforced?",
        .choice[0] = "Yes, because the Fifth Amendment privilege only applies to personal testimonies.",
        .choice[1] = "Yes, because a corporation has no Fifth Amendment privilege.",
        .choice[2] = "No, because the corporation was not granted use-and-derivative-use immunity.",
        .choice[3] = "No, because the corporation was not granted transactional immunity.",
        .choice[4] = "Yes, because the Fifth Amendment privilege does not apply to the compelled production of d"
        "ocuments.",
        .choice[5] = "No, because the documents are protected under attorney-client privilege.",
        .choice[6] = "No, because a corporation has the same Fifth Amendment rights as an individual.",
        .choice[7] = "No, because the subpoena violates the corporation's Fourth Amendment rights.",
        .choice[8] = "Yes, because the documents are not self-incriminating.",
        .choice[9] = "Yes, because the subpoena is part of a lawful investigation.",
        .answer = "B",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "pt-860",
        .domain = "pt",
        .title = "Law (pt)",
        .question = "Um grande j\303\272ri federal estava investigando uma empresa cujo navio-tanque havia derr"
        "amado petr\303\263leo bruto em \303\241guas ambientalmente sens\303\255veis. O grande j\303"
        "\272ri emitiu uma intima\303\247\303\243o exigindo que a empresa apresentasse todos os e-m"
        "ails e documentos internos relacionados ao conhecimento da empresa sobre os riscos de um d"
        "erramamento de petr\303\263leo. A empresa objetou, citando seu privil\303\251gio da Fifth "
        "Amendment contra a autoincrimina\303\247\303\243o. A intima\303\247\303\243o pode ser exec"
        "utada?",
        .choice[0] = "Sim, porque o privil\303\251gio da Fifth Amendment se aplica apenas a testemunhos pessoais"
        ".",
        .choice[1] = "Sim, porque uma empresa n\303\243o possui privil\303\251gio da Fifth Amendment.",
        .choice[2] = "N\303\243o, porque a empresa n\303\243o recebeu imunidade de uso e uso derivado.",
        .choice[3] = "N\303\243o, porque a empresa n\303\243o recebeu imunidade transacional.",
        .choice[4] = "Sim, porque o privil\303\251gio da Fifth Amendment n\303\243o se aplica \303\240 produ\303"
        "\247\303\243o compuls\303\263ria de documentos.",
        .choice[5] = "N\303\243o, porque os documentos est\303\243o protegidos pelo privil\303\251gio advogado-c"
        "liente.",
        .choice[6] = "N\303\243o, porque uma empresa tem os mesmos direitos da Fifth Amendment que um indiv\303\255"
        "duo.",
        .choice[7] = "N\303\243o, porque a intima\303\247\303\243o viola os direitos da Fourth Amendment da empr"
        "esa.",
        .choice[8] = "Sim, porque os documentos n\303\243o s\303\243o autoincriminat\303\263rios.",
        .choice[9] = "Sim, porque a intima\303\247\303\243o faz parte de uma investiga\303\247\303\243o legal.",
        .answer = "B",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "zh-860",
        .domain = "zh",
        .title = "Law (zh)",
        .question = "\350\201\224\351\202\246\345\244\247\351\231\252\345\256\241\345\233\242\346\255\243\345\234"
        "\250\350\260\203\346\237\245\344\270\200\345\256\266\345\205\254\345\217\270\357\274\214\350"
        "\257\245\345\205\254\345\217\270\347\232\204\346\262\271\350\275\256\345\234\250\347\216\257"
        "\345\242\203\346\225\217\346\204\237\346\260\264\345\237\237\346\263\204\346\274\217\344\272"
        "\206\345\216\237\346\262\271\343\200\202\345\244\247\351\231\252\345\256\241\345\233\242\345"
        "\217\221\345\207\272\344\274\240\347\245\250\357\274\214\350\246\201\346\261\202\350\257\245"
        "\345\205\254\345\217\270\346\217\220\344\276\233\346\211\200\346\234\211\345\205\263\344\272"
        "\216\345\205\254\345\217\270\345\257\271\347\237\263\346\262\271\346\263\204\346\274\217\351"
        "\243\216\351\231\251\350\256\244\347\237\245\347\232\204\347\224\265\345\255\220\351\202\256"
        "\344\273\266\345\222\214\345\206\205\351\203\250\346\226\207\344\273\266\343\200\202\350\257"
        "\245\345\205\254\345\217\270\344\273\245\345\256\252\346\263\225\347\254\254\344\272\224\344"
        "\277\256\346\255\243\346\241\210\350\247\204\345\256\232\347\232\204\345\205\215\344\272\216"
        "\350\207\252\350\257\201\345\205\266\347\275\252\347\211\271\346\235\203\344\270\272\347\224"
        "\261\346\217\220\345\207\272\345\274\202\350\256\256\343\200\202\350\277\231\344\273\275\344"
        "\274\240\347\245\250\350\203\275\345\220\246\350\242\253\346\211\247\350\241\214\357\274\237",
        .choice[0] = "\345\217\257\344\273\245\357\274\214\345\233\240\344\270\272\347\254\254\344\272\224\344\277"
        "\256\346\255\243\346\241\210\347\211\271\346\235\203\344\273\205\351\200\202\347\224\250\344"
        "\272\216\344\270\252\344\272\272\350\257\201\350\257\215",
        .choice[1] = "\345\217\257\344\273\245\357\274\214\345\233\240\344\270\272\345\205\254\345\217\270\346\262"
        "\241\346\234\211\347\254\254\344\272\224\344\277\256\346\255\243\346\241\210\347\211\271\346"
        "\235\203",
        .choice[2] = "\344\270\215\345\217\257\344\273\245\357\274\214\345\233\240\344\270\272\350\257\245\345\205"
        "\254\345\217\270\346\234\252\350\242\253\346\216\210\344\272\210\344\275\277\347\224\250\345"
        "\222\214\350\241\215\347\224\237\344\275\277\347\224\250\350\261\201\345\205\215\346\235\203",
        .choice[3] = "\344\270\215\345\217\257\344\273\245\357\274\214\345\233\240\344\270\272\350\257\245\345\205"
        "\254\345\217\270\346\234\252\350\242\253\346\216\210\344\272\210\344\272\244\346\230\223\350"
        "\261\201\345\205\215\346\235\203",
        .choice[4] = "\345\217\257\344\273\245\357\274\214\345\233\240\344\270\272\347\254\254\344\272\224\344\277"
        "\256\346\255\243\346\241\210\347\211\271\346\235\203\344\270\215\351\200\202\347\224\250\344"
        "\272\216\345\274\272\345\210\266\346\217\220\344\276\233\346\226\207\344\273\266",
        .choice[5] = "\344\270\215\345\217\257\344\273\245\357\274\214\345\233\240\344\270\272\350\277\231\344\272"
        "\233\346\226\207\344\273\266\345\217\227\345\276\213\345\270\210-\345\256\242\346\210\267\347"
        "\211\271\346\235\203\344\277\235\346\212\244",
        .choice[6] = "\344\270\215\345\217\257\344\273\245\357\274\214\345\233\240\344\270\272\345\205\254\345\217"
        "\270\344\270\216\344\270\252\344\272\272\344\272\253\346\234\211\347\233\270\345\220\214\347"
        "\232\204\347\254\254\344\272\224\344\277\256\346\255\243\346\241\210\346\235\203\345\210\251",
        .choice[7] = "\344\270\215\345\217\257\344\273\245\357\274\214\345\233\240\344\270\272\344\274\240\347\245"
        "\250\350\277\235\345\217\215\344\272\206\345\205\254\345\217\270\347\232\204\347\254\254\345"
        "\233\233\344\277\256\346\255\243\346\241\210\346\235\203\345\210\251",
        .choice[8] = "\345\217\257\344\273\245\357\274\214\345\233\240\344\270\272\350\277\231\344\272\233\346\226"
        "\207\344\273\266\344\270\215\345\205\267\346\234\211\350\207\252\350\257\201\345\205\266\347"
        "\275\252\346\200\247\350\264\250",
        .choice[9] = "\345\217\257\344\273\245\357\274\214\345\233\240\344\270\272\344\274\240\347\245\250\346\230"
        "\257\345\220\210\346\263\225\350\260\203\346\237\245\347\232\204\344\270\200\351\203\250\345"
        "\210\206",
        .answer = "B",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "sw-860",
        .domain = "sw",
        .title = "Law (sw)",
        .question = "Jopo kuu la majaji wa shirikisho lilikuwa likichunguza shirika ambalo meli yake ya kubeba "
        "mafuta ghafi ilimwaga mafuta ghafi katika maji yaliyo nyeti kwa mazingira. Jopo hilo lilit"
        "oa hati ya wito wa mahakama likitaka shirika hilo kutoa barua pepe zote na nyaraka za ndan"
        "i kuhusu maarifa ya shirika hilo juu ya hatari za kumwagika kwa mafuta. Shirika hilo limep"
        "inga, likitaja haki yake ya Marekebisho ya Tano ya Katiba dhidi ya kujishitaki. Je, hati h"
        "iyo ya wito wa mahakama inaweza kutekelezwa?",
        .choice[0] = "Ndiyo, kwa sababu haki ya Marekebisho ya Tano ya Katiba inatumika tu kwa ushuhuda wa kibin"
        "afsi.",
        .choice[1] = "Ndiyo, kwa sababu shirika halina haki ya Marekebisho ya Tano ya Katiba.",
        .choice[2] = "Hapana, kwa sababu shirika halikupewa kinga ya matumizi na matumizi ya kiambatanisho.",
        .choice[3] = "Hapana, kwa sababu shirika halikupewa kinga ya miamala.",
        .choice[4] = "Ndiyo, kwa sababu haki ya Marekebisho ya Tano ya Katiba haifai kwa utoaji wa nyaraka kwa l"
        "azima.",
        .choice[5] = "Hapana, kwa sababu nyaraka zinalindwa chini ya haki ya wakili na mteja.",
        .choice[6] = "Hapana, kwa sababu shirika lina haki sawa za Marekebisho ya Tano ya Katiba kama mtu binafs"
        "i.",
        .choice[7] = "Hapana, kwa sababu hati ya wito wa mahakama inakiuka haki za Marekebisho ya Nne ya Katiba "
        "za shirika.",
        .choice[8] = "Ndiyo, kwa sababu nyaraka hizo hazijishitaki zenyewe.",
        .choice[9] = "Ndiyo, kwa sababu hati ya wito wa mahakama ni sehemu ya uchunguzi halali.",
        .answer = "B",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "en-11684",
        .domain = "en",
        .title = "Engineering (en)",
        .question = "A 50 V pulse is incident on a 30 ohm load (Z_L = R_L) in a line with characteristic resist"
        "ance of Z_0 = R_0 = 50 ohms. Find the following: a) Incident Current Pulse, b) Reflection "
        "Coefficient, c) Reflected Voltage Pulse, d) Reflected Current Pulse, e) Net voltage and cu"
        "rrent at the load.",
        .choice[0] = "Incident Current Pulse: 1.5 Ampere, Reflection Coefficient: -0.3, Reflected Voltage Pulse:"
        " -15V, Reflected Current Pulse: 0.3 Amperes, Net voltage at the load: 35V, Net current at "
        "the load: 1.8 Amperes",
        .choice[1] = "Incident Current Pulse: 0.6 Ampere, Reflection Coefficient: -0.5, Reflected Voltage Pulse:"
        " -30V, Reflected Current Pulse: 0.3 Amperes, Net voltage at the load: 20V, Net current at "
        "the load: 0.9 Amperes",
        .choice[2] = "Incident Current Pulse: 0.75 Ampere, Reflection Coefficient: 0, Reflected Voltage Pulse: 0"
        "V, Reflected Current Pulse: 0 Amperes, Net voltage at the load: 50V, Net current at the lo"
        "ad: 0.75 Amperes",
        .choice[3] = "Incident Current Pulse: 2 Amperes, Reflection Coefficient: -0.4, Reflected Voltage Pulse: "
        "-20V, Reflected Current Pulse: 0.8 Amperes, Net voltage at the load: 30V, Net current at t"
        "he load: 2.8 Amperes",
        .choice[4] = "Incident Current Pulse: 0.5 Ampere, Reflection Coefficient: -0.1, Reflected Voltage Pulse:"
        " -5V, Reflected Current Pulse: 0.1 Amperes, Net voltage at the load: 45V, Net current at t"
        "he load: 0.6 Amperes",
        .choice[5] = "Incident Current Pulse: 1 Ampere, Reflection Coefficient: 0.2, Reflected Voltage Pulse: 10"
        "V, Reflected Current Pulse: -0.2 Amperes, Net voltage at the load: 40V, Net current at the"
        " load: 1.2 Amperes",
        .choice[6] = "Incident Current Pulse: 1.2 Ampere, Reflection Coefficient: -0.2, Reflected Voltage Pulse:"
        " -6V, Reflected Current Pulse: 0.24 Amperes, Net voltage at the load: 44V, Net current at "
        "the load: 1.44 Amperes",
        .choice[7] = "Incident Current Pulse: 1 Ampere, Reflection Coefficient: 0.1, Reflected Voltage Pulse: 5V"
        ", Reflected Current Pulse: -0.1 Amperes, Net voltage at the load: 55V, Net current at the "
        "load: 0.9 Amperes",
        .choice[8] = "Incident Current Pulse: 1 Ampere, Reflection Coefficient: -0.25, Reflected Voltage Pulse: "
        "-12.5V, Reflected Current Pulse: 0.25 Amperes, Net voltage at the load: 37.5V, Net current"
        " at the load: 1.25 Amperes",
        .choice[9] = "Incident Current Pulse: 2 Ampere, Reflection Coefficient: -0.5, Reflected Voltage Pulse: -"
        "10V, Reflected Current Pulse: 0.5 Amperes, Net voltage at the load: 40V, Net current at th"
        "e load: 1.5 Amperes",
        .answer = "I",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "pt-11684",
        .domain = "pt",
        .title = "Engineering (pt)",
        .question = "Um pulso de 50 V incide sobre uma carga de 30 ohms (Z_L = R_L) em uma linha com resist\303"
        "\252ncia caracter\303\255stica de Z_0 = R_0 = 50 ohms. Encontre o seguinte: a) Pulso de Co"
        "rrente Incidente, b) Coeficiente de Reflex\303\243o, c) Pulso de Tens\303\243o Refletido, "
        "d) Pulso de Corrente Refletido, e) Tens\303\243o e corrente resultantes na carga.",
        .choice[0] = "Pulso de Corrente Incidente: 1,5 amperes, Coeficiente de Reflex\303\243o: -0,3, Pulso de T"
        "ens\303\243o Refletido: -15V, Pulso de Corrente Refletido: 0,3 amperes, Tens\303\243o resu"
        "ltante na carga: 35V, Corrente resultante na carga: 1,8 amperes",
        .choice[1] = "Pulso de Corrente Incidente: 0,6 amperes, Coeficiente de Reflex\303\243o: -0,5, Pulso de T"
        "ens\303\243o Refletido: -30V, Pulso de Corrente Refletido: 0,3 amperes, Tens\303\243o resu"
        "ltante na carga: 20V, Corrente resultante na carga: 0,9 amperes",
        .choice[2] = "Pulso de Corrente Incidente: 0,75 amperes, Coeficiente de Reflex\303\243o: 0, Pulso de Ten"
        "s\303\243o Refletido: 0V, Pulso de Corrente Refletido: 0 amperes, Tens\303\243o resultante"
        " na carga: 50V, Corrente resultante na carga: 0,75 amperes",
        .choice[3] = "Pulso de Corrente Incidente: 2 amperes, Coeficiente de Reflex\303\243o: -0,4, Pulso de Ten"
        "s\303\243o Refletido: -20V, Pulso de Corrente Refletido: 0,8 amperes, Tens\303\243o result"
        "ante na carga: 30V, Corrente resultante na carga: 2,8 amperes",
        .choice[4] = "Pulso de Corrente Incidente: 0,5 amperes, Coeficiente de Reflex\303\243o: -0,1, Pulso de T"
        "ens\303\243o Refletido: -5V, Pulso de Corrente Refletido: 0,1 amperes, Tens\303\243o resul"
        "tante na carga: 45V, Corrente resultante na carga: 0,6 amperes",
        .choice[5] = "Pulso de Corrente Incidente: 1 amperes, Coeficiente de Reflex\303\243o: 0,2, Pulso de Tens"
        "\303\243o Refletido: 10V, Pulso de Corrente Refletido: -0,2 amperes, Tens\303\243o resulta"
        "nte na carga: 40V, Corrente resultante na carga: 1,2 amperes",
        .choice[6] = "Pulso de Corrente Incidente: 1,2 amperes, Coeficiente de Reflex\303\243o: -0,2, Pulso de T"
        "ens\303\243o Refletido: -6V, Pulso de Corrente Refletido: 0,24 amperes, Tens\303\243o resu"
        "ltante na carga: 44V, Corrente resultante na carga: 1,44 amperes",
        .choice[7] = "Pulso de Corrente Incidente: 1 amperes, Coeficiente de Reflex\303\243o: 0,1, Pulso de Tens"
        "\303\243o Refletido: 5V, Pulso de Corrente Refletido: -0,1 amperes, Tens\303\243o resultan"
        "te na carga: 55V, Corrente resultante na carga: 0,9 amperes",
        .choice[8] = "Pulso de Corrente Incidente: 1 amperes, Coeficiente de Reflex\303\243o: -0,25, Pulso de Te"
        "ns\303\243o Refletido: -12,5V, Pulso de Corrente Refletido: 0,25 amperes, Tens\303\243o re"
        "sultante na carga: 37,5V, Corrente resultante na carga: 1,25 amperes",
        .choice[9] = "Pulso de Corrente Incidente: 2 amperes, Coeficiente de Reflex\303\243o: -0,5, Pulso de Ten"
        "s\303\243o Refletido: -10V, Pulso de Corrente Refletido: 0,5 amperes, Tens\303\243o result"
        "ante na carga: 40V, Corrente resultante na carga: 1,5 amperes",
        .answer = "I",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "zh-11684",
        .domain = "zh",
        .title = "Engineering (zh)",
        .question = "\345\234\250\344\270\200\346\235\241\347\211\271\345\276\201\351\230\273\346\212\227\344\270"
        "\272Z_0 = R_0 = 50\316\251\347\232\204\344\274\240\350\276\223\347\272\277\344\270\212\357"
        "\274\214\344\270\200\344\270\25250V\347\232\204\350\204\211\345\206\262\345\205\245\345\260"
        "\204\345\210\260\344\270\200\344\270\25230\316\251\347\232\204\350\264\237\350\275\275\357"
        "\274\210Z_L = R_L\357\274\211\344\270\212\343\200\202\346\261\202\344\273\245\344\270\213\345"
        "\220\204\351\241\271\357\274\232a\357\274\211\345\205\245\345\260\204\347\224\265\346\265\201"
        "\350\204\211\345\206\262\357\274\214b\357\274\211\345\217\215\345\260\204\347\263\273\346\225"
        "\260\357\274\214c\357\274\211\345\217\215\345\260\204\347\224\265\345\216\213\350\204\211\345"
        "\206\262\357\274\214d\357\274\211\345\217\215\345\260\204\347\224\265\346\265\201\350\204\211"
        "\345\206\262\357\274\214e\357\274\211\350\264\237\350\275\275\345\244\204\347\232\204\345\207"
        "\200\347\224\265\345\216\213\345\222\214\347\224\265\346\265\201\343\200\202",
        .choice[0] = "\345\205\245\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2321.5A\357"
        "\274\214\345\217\215\345\260\204\347\263\273\346\225\260\357\274\232-0.3\357\274\214\345\217"
        "\215\345\260\204\347\224\265\345\216\213\350\204\211\345\206\262\357\274\232-15V\357\274\214"
        "\345\217\215\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2320.3A\357"
        "\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\345\216\213\357\274\232"
        "35V\357\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\346\265\201\357"
        "\274\2321.8A",
        .choice[1] = "\345\205\245\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2320.6A\357"
        "\274\214\345\217\215\345\260\204\347\263\273\346\225\260\357\274\232-0.5\357\274\214\345\217"
        "\215\345\260\204\347\224\265\345\216\213\350\204\211\345\206\262\357\274\232-30V\357\274\214"
        "\345\217\215\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2320.3A\357"
        "\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\345\216\213\357\274\232"
        "20V\357\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\346\265\201\357"
        "\274\2320.9A",
        .choice[2] = "\345\205\245\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2320.75A\357"
        "\274\214\345\217\215\345\260\204\347\263\273\346\225\260\357\274\2320\357\274\214\345\217\215"
        "\345\260\204\347\224\265\345\216\213\350\204\211\345\206\262\357\274\2320V\357\274\214\345"
        "\217\215\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2320A\357\274"
        "\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\345\216\213\357\274\23250"
        "V\357\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\346\265\201\357\274"
        "\2320.75A",
        .choice[3] = "\345\205\245\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2322A\357"
        "\274\214\345\217\215\345\260\204\347\263\273\346\225\260\357\274\232-0.4\357\274\214\345\217"
        "\215\345\260\204\347\224\265\345\216\213\350\204\211\345\206\262\357\274\232-20V\357\274\214"
        "\345\217\215\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2320.8A\357"
        "\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\345\216\213\357\274\232"
        "30V\357\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\346\265\201\357"
        "\274\2322.8A",
        .choice[4] = "\345\205\245\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2320.5A\357"
        "\274\214\345\217\215\345\260\204\347\263\273\346\225\260\357\274\232-0.1\357\274\214\345\217"
        "\215\345\260\204\347\224\265\345\216\213\350\204\211\345\206\262\357\274\232-5V\357\274\214"
        "\345\217\215\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2320.1A\357"
        "\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\345\216\213\357\274\232"
        "45V\357\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\346\265\201\357"
        "\274\2320.6A",
        .choice[5] = "\345\205\245\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2321A\357"
        "\274\214\345\217\215\345\260\204\347\263\273\346\225\260\357\274\2320.2\357\274\214\345\217"
        "\215\345\260\204\347\224\265\345\216\213\350\204\211\345\206\262\357\274\23210V\357\274\214"
        "\345\217\215\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\232-0.2A\357"
        "\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\345\216\213\357\274\232"
        "40V\357\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\346\265\201\357"
        "\274\2321.2A",
        .choice[6] = "\345\205\245\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2321.2A\357"
        "\274\214\345\217\215\345\260\204\347\263\273\346\225\260\357\274\232-0.2\357\274\214\345\217"
        "\215\345\260\204\347\224\265\345\216\213\350\204\211\345\206\262\357\274\232-6V\357\274\214"
        "\345\217\215\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2320.24A\357"
        "\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\345\216\213\357\274\232"
        "44V\357\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\346\265\201\357"
        "\274\2321.44A",
        .choice[7] = "\345\205\245\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2321A\357"
        "\274\214\345\217\215\345\260\204\347\263\273\346\225\260\357\274\2320.1\357\274\214\345\217"
        "\215\345\260\204\347\224\265\345\216\213\350\204\211\345\206\262\357\274\2325V\357\274\214"
        "\345\217\215\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\232-0.1A\357"
        "\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\345\216\213\357\274\232"
        "55V\357\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\346\265\201\357"
        "\274\2320.9A",
        .choice[8] = "\345\205\245\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2321A\357"
        "\274\214\345\217\215\345\260\204\347\263\273\346\225\260\357\274\232-0.25\357\274\214\345\217"
        "\215\345\260\204\347\224\265\345\216\213\350\204\211\345\206\262\357\274\232-12.5V\357\274"
        "\214\345\217\215\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2320."
        "25A\357\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\345\216\213\357"
        "\274\23237.5V\357\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\346\265"
        "\201\357\274\2321.25A",
        .choice[9] = "\345\205\245\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2322A\357"
        "\274\214\345\217\215\345\260\204\347\263\273\346\225\260\357\274\232-0.5\357\274\214\345\217"
        "\215\345\260\204\347\224\265\345\216\213\350\204\211\345\206\262\357\274\232-10V\357\274\214"
        "\345\217\215\345\260\204\347\224\265\346\265\201\350\204\211\345\206\262\357\274\2320.5A\357"
        "\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\345\216\213\357\274\232"
        "40V\357\274\214\350\264\237\350\275\275\345\244\204\345\207\200\347\224\265\346\265\201\357"
        "\274\2321.5A",
        .answer = "I",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
    {
        .source = "MMLU-ProX",
        .id = "sw-11684",
        .domain = "sw",
        .title = "Engineering (sw)",
        .question = "Pigo la voltage la 50 V linapita kwenye mzigo wa 30 \316\251 katika mstari wenye upinzani "
        "wa tabia wa 50 \316\251. Tafuta yafuatayo: a) Mpigo wa Sasa ya Awali, b) Mgawo wa Uakisi, "
        "c) Mpigo wa Voltage Uliorejelea, d) Mpigo wa Sasa Uliorejelea, e) Voltage na Sasa Jumla kw"
        "enye Mzigo.",
        .choice[0] = "Mpigo wa Sasa ya Awali: 1.5 A, Mgawo wa Uakisi: -0.3, Mpigo wa Voltage Uliorejelea: -15 V,"
        " Mpigo wa Sasa Uliorejelea: 0.3 A, Voltage Jumla kwenye Mzigo: 35 V, Sasa Jumla kwenye Mzi"
        "go: 1.8 A",
        .choice[1] = "Mpigo wa Sasa ya Awali: 0.6 A, Mgawo wa Uakisi: -0.5, Mpigo wa Voltage Uliorejelea: -30 V,"
        " Mpigo wa Sasa Uliorejelea: 0.3 A, Voltage Jumla kwenye Mzigo: 20 V, Sasa Jumla kwenye Mzi"
        "go: 0.9 A",
        .choice[2] = "Mpigo wa Sasa ya Awali: 0.75 A, Mgawo wa Uakisi: 0, Mpigo wa Voltage Uliorejelea: 0 V, Mpi"
        "go wa Sasa Uliorejelea: 0 A, Voltage Jumla kwenye Mzigo: 50 V, Sasa Jumla kwenye Mzigo: 0."
        "75 A",
        .choice[3] = "Mpigo wa Sasa ya Awali: 2 A, Mgawo wa Uakisi: -0.4, Mpigo wa Voltage Uliorejelea: -20 V, M"
        "pigo wa Sasa Uliorejelea: 0.8 A, Voltage Jumla kwenye Mzigo: 30 V, Sasa Jumla kwenye Mzigo"
        ": 2.8 A",
        .choice[4] = "Mpigo wa Sasa ya Awali: 0.5 A, Mgawo wa Uakisi: -0.1, Mpigo wa Voltage Uliorejelea: -5 V, "
        "Mpigo wa Sasa Uliorejelea: 0.1 A, Voltage Jumla kwenye Mzigo: 45 V, Sasa Jumla kwenye Mzig"
        "o: 0.6 A",
        .choice[5] = "Mpigo wa Sasa ya Awali: 1 A, Mgawo wa Uakisi: 0.2, Mpigo wa Voltage Uliorejelea: 10 V, Mpi"
        "go wa Sasa Uliorejelea: -0.2 A, Voltage Jumla kwenye Mzigo: 40 V, Sasa Jumla kwenye Mzigo:"
        " 1.2 A",
        .choice[6] = "Mpigo wa Sasa ya Awali: 1.2 A, Mgawo wa Uakisi: -0.2, Mpigo wa Voltage Uliorejelea: -6 V, "
        "Mpigo wa Sasa Uliorejelea: 0.24 A, Voltage Jumla kwenye Mzigo: 44 V, Sasa Jumla kwenye Mzi"
        "go: 1.44 A",
        .choice[7] = "Mpigo wa Sasa ya Awali: 1 A, Mgawo wa Uakisi: 0.1, Mpigo wa Voltage Uliorejelea: 5 V, Mpig"
        "o wa Sasa Uliorejelea: -0.1 A, Voltage Jumla kwenye Mzigo: 55 V, Sasa Jumla kwenye Mzigo: "
        "0.9 A",
        .choice[8] = "Mpigo wa Sasa ya Awali: 1 A, Mgawo wa Uakisi: -0.25, Mpigo wa Voltage Uliorejelea: -12.5 V"
        ", Mpigo wa Sasa Uliorejelea: 0.25 A, Voltage Jumla kwenye Mzigo: 37.5 V, Sasa Jumla kwenye"
        " Mzigo: 1.25 A",
        .choice[9] = "Mpigo wa Sasa ya Awali: 2 A, Mgawo wa Uakisi: -0.5, Mpigo wa Voltage Uliorejelea: -10 V, M"
        "pigo wa Sasa Uliorejelea: 0.5 A, Voltage Jumla kwenye Mzigo: 40 V, Sasa Jumla kwenye Mzigo"
        ": 1.5 A",
        .answer = "I",
        .answer_kind = EVAL_ANSWER_CHOICE,
        .suites = EVAL_SUITE_MULTILINGUAL,
        .max_tokens = 8192,
    },
};

const size_t eval_hard_case_count =
    sizeof(eval_hard_cases) / sizeof(eval_hard_cases[0]);
