# Mentor Session Prep — 3:30 PM

Read this once. Know it cold. You have 50 minutes.

---

## 1. The Framing Line

> "RecallTrace is a benchmark where the agent sees a contamination pattern in a partially observable graph — and has to figure out which hidden causal intervention produced it, using tool calls and a calibrated belief state, before it decides what to quarantine."

That's the line. Say it first. It immediately separates you from every team that built a game or a logistics optimizer. The words "hidden causal intervention" and "partial evidence" are doing the work — they tell a Meta engineer this is an inference problem, not a planning problem.

If the mentor looks interested, follow with: "And we added an adversary that makes the problem harder as the agent improves — so the benchmark evolves with the agent."

---

## 2. The Hard-Case Scenario

Use this when someone says "give me an example."

> **"Here's what a hard episode looks like."**
>
> Lot A is contaminated at a warehouse. It gets repacked into Lot B and Lot C, and Lot C gets mixed with safe stock at a crossdock. Then the record of that repack is deleted — so when the agent inspects the crossdock, it sees partial contamination but no paper trail connecting it back to Lot A.
>
> The agent has to figure out that the contamination at the crossdock didn't originate there — it came through a hidden relabel hop whose record was deleted. It does this by cross-referencing lot origins and noticing that Lot C's creation timestamp matches Lot A's repack window, even though there's no explicit link.
>
> The correct action is to quarantine Lot B and the contaminated portion of Lot C, but leave the safe stock at the crossdock alone. An untrained agent quarantines the entire crossdock — six lots instead of two. A trained agent quarantines exactly two.

That's three sentences of setup, three of reasoning. Stop there. Let them ask follow-ups.

---

## 3. Four Mentor Questions

### Question 1: Reward Design Validation

**Say this:**
> "Our reward has three components — recall, precision, and a calibration bonus. The calibration bonus gives +0.3 if the agent's belief exceeds 0.8 before it quarantines. Is that the right way to incentivize well-calibrated confidence, or should we be penalizing miscalibrated quarantines instead?"

**Why this matters:**
This decides whether we keep the current reward or restructure it before Round 2. If the mentor says "penalize miscalibration," we flip the sign and retrain. If they say "bonus is fine," we lock the reward and move on.

**Good answer:** "The bonus approach is fine, but consider scaling it — 0.3 might be too weak relative to the +2.0 recall signal." → Action: tune the coefficient, don't restructure.

**Bad answer:** "I'd need to see the training curves to say." → They're not engaging with the design. Move to the next question.

---

### Question 2: Self-Play Framing

**Say this:**
> "We have an adversary that chooses where to hide the intervention — and it learns which placements make the investigator fail. A mentor at another hackathon told us static curricula are fine and self-play is overkill for benchmarks. Do you think the adversary adds genuine value here, or should we have spent that time on a better base environment?"

**Why this matters:**
This is the biggest bet in the project. If the mentor validates self-play, you double down on it for the final pitch. If they push back, you know to lead with the causal inference framing and treat self-play as a secondary feature.

**Good answer:** "The adversary is interesting because it gives you an automatic difficulty curriculum — that's a real contribution." → Lead with self-play in the final pitch.

**Bad answer:** "Self-play is cool but judges care more about the environment quality itself." → Lead with causal inference, mention self-play as a bonus.

---

### Question 3: Theme Alignment Check

**Say this:**
> "We're positioning this as Theme 3.1 — world modeling — because the agent maintains a belief state and does causal reasoning. But we also hit Theme 4 — self-play and recursive skill amplification. Should we pick one primary theme and go deep, or is it stronger to show we hit both?"

**Why this matters:**
This decides your final slide structure. One-theme means a focused 3-minute pitch. Two-theme means you need to show both are load-bearing, which is harder but more impressive if you pull it off.

**Good answer:** "If both are genuine, show both — judges remember submissions that hit multiple themes." → Keep the dual-theme pitch.

**Bad answer:** "Pick one and go deep. Judges get confused when you try to hit everything." → Cut Theme 4 from the opening, mention it once at the end.

---

### Question 4: What's Missing

**Say this:**
> "We have the environment, the self-play loop, training curves, and a before/after demo. If you were judging this submission, what's the one thing you'd want to see that we don't have yet?"

**Why this matters:**
This is the question that gets you the most value per second. The mentor tells you exactly what to build between now and 8 PM. Whatever they say, build it.

**Good answer:** Anything specific — "show me the belief state updating in real time," "I want to see what happens when you increase graph size," "add a comparison to a random baseline." → Build exactly that.

**Bad answer:** "Looks good, nothing comes to mind." → They're being polite. Ask: "If you had to cut one thing from the final pitch, what would you cut?" That forces a real answer.

---

## 4. The Closing Line

**Say this at the end of the session:**

> "For Round 2 at 8 PM, we're going to show the full training loop running live — reset, episode, belief tracker updating, F1 climbing. If there's one thing you want us to make sure is in that demo, what is it?"

This does three things: it tells them you have a plan, it gives them a specific time to see you again, and it gets you one more piece of actionable feedback on the way out.

---

## Session Flow — 10 Minutes Total

| Time | What you do |
|---|---|
| 0:00–0:30 | Say the framing line. Open the architecture diagram on your laptop. |
| 0:30–1:30 | Walk through the hard-case scenario (Lot A → B+C). |
| 1:30–2:00 | Show `before_after_demo.png` — "this is the agent before and after training." |
| 2:00–8:00 | Ask the 4 questions. Listen. Take notes. |
| 8:00–9:00 | Ask the closing line. Write down whatever they say. |
| 9:00–10:00 | Thank them. Close laptop. Move to next mentor. |

---

## If You Get Nervous

Look at the architecture diagram on your screen. Point at Layer 2 and say: "This is the part that makes it causal — the agent doesn't know which intervention happened. It has to figure it out." Then point at Layer 6 and say: "And this is the part that makes it self-improving — the adversary makes the problem harder as the agent gets smarter."

That's it. Two layers. Two sentences. Everything else is follow-up.

Go get it, Shamanth.
