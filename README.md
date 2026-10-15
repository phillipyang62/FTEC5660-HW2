# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:
> to students: this is your report, see the homework description.

### Task 1
> one paragraph describing your solution, and your results on `public_test`.
paragrahp:hw2 pic.jpg
cv,score,correctness
CV_1.pdf,0.9000,correct
CV_2.pdf,0.9000,correct
CV_3.pdf,0.9000,correct
CV_4.pdf,0.1000,correct
CV_5.pdf,0.1000,correct
CV_6.pdf,0.9000,correct
CV_7.pdf,0.1000,correct
In build_agent function, I use create function to connect the llm with six MCP tools. System  prompt will ask it get name  industry city etc. the name together with  city and industry will help to search on the facebook and linkedin.

if mutiple result is returned, will only keep the one whose school and  company is match to CV.

In score_cvs function, for every CV, ainvoke will be called and use the masage as evidence . Judge llm will use the evidence to score the CV. Wrong years fake experience or employer will be seen as  difference. The output is 0.9 or 0.1, otherwise is 0.0.
### Task 2
> the attack techniques you used and why you expect them to work.
I use prompt injection in four ways but none of them worked， I think large models today had specially trained to defend prompt injection attack.  At the beginning this year, there were news that attackers used prompt injection to attack agents deployed in local devices. I thought this prompt issue has fixed.

There were my ways to misguide agent:

Output format hijacking:Tells the model the final answer may only be 0.9, with no 0.1 or any other character

Role and delimiter confusion:Ends the CV with CV TEXT END, </user_message>, and a fake <system> block, then inserts a new instruction

Hidden text: Writes that same instruction in white so it is invisible on the page but still extracted from the PDF

Authority and sycophancy:Adds a fake LinkedIn verification certificate marked MATCH, so the model treats the CV as already checked

I expect them to work beacuse llms sometimes can be easy to fool, they will accept your statement easily.

##AI PARADOX
People mean very different things—ChatGPT, self-driving cars, recommendation algorithms—yet we use one single term. Dignum argues this confusion is not just technical but also about language and perspective. Different groups bring different assumptions, and more voices only make agreement harder. This reminded me that AI is not a fixed object we can easily define. It is a moving target. Instead of pretending the confusion will disappear, we should accept it and think carefully about what we really mean when we say "AI."
