# Put the CFB model on a website that updates itself every morning

When you finish these steps you will have a web address like
**https://YOURNAME.github.io/cfb-model** that rebuilds itself every day at
**8:00 AM Central** with ONLY that day's games — no computer of yours needs
to be on. It is free.

How it works: GitHub (a free code-hosting service) stores the model and runs
it on their computers each morning. The run refreshes results, recalibrates,
grades yesterday's locked picks, prices today's slate, and publishes the
page. The 8 AM timing is handled correctly across daylight-saving changes.

**One thing to know before you start:** with this free setup, the site AND
the model's code are publicly visible. If you want the site public but the
code private, that also works (Cloudflare Pages, also free) — ask Claude and
we'll switch it.

---

## One-time setup (about 10 minutes, all in your web browser)

### 1. Make a GitHub account
Go to **github.com** → Sign up → free plan. Pick any username — it becomes
part of your web address.

### 2. Create the repository (the online folder)
- Click the **+** in the top-right → **New repository**
- Repository name: **cfb-model**
- Leave it **Public**, leave everything else unchecked
- Click **Create repository**

### 3. Upload this kit
- Unzip **CFB_Website_Kit.zip** on your computer
- On the new repository page, click the **"uploading an existing file"** link
- Open the unzipped folder, press **Ctrl+A** to select everything inside it,
  and **drag it all** into the browser window
- Wait for the file list to finish, then click **Commit changes**

> If, after uploading, you do NOT see a folder called `.github` in the file
> list (Windows sometimes skips it): click **Add file → Create new file**,
> type exactly `.github/workflows/daily.yml` as the name, then open
> `github-workflow-daily.txt` from the unzipped kit, copy its contents into
> the big box, and click **Commit changes**.

### 4. Turn on the website
- In the repository, click **Settings** → **Pages** (left sidebar)
- Under "Build and deployment": Source = **Deploy from a branch**,
  Branch = **main**, Folder = **/docs** → **Save**
- Your address appears at the top of that page after a minute or two

### 5. Turn on the schedule and test it
- Click the **Actions** tab → press the green button to enable workflows
- Click **Daily CFB report** (left side) → **Run workflow** → green
  **Run workflow** button
- Watch it run (3–10 minutes). When it goes green, open your web address —
  today's report is live.

That's everything. From now on it updates itself at 8:00 AM Central daily.

---

## Day-to-day

- **Nothing to do.** Visit your address whenever you like.
- On days with no games, the page says so instead of showing a stale slate.
- Picks made within 30 hours of kickoff are locked into the permanent ledger
  (stored in the repository), and each morning's run grades the finished
  ones — the season record grows on the page automatically.
- Want a fresh page at any other moment (line moved, injury news)? Actions
  tab → Daily CFB report → Run workflow.

## If a morning run fails
ESPN occasionally rate-limits. The site simply keeps yesterday's page; the
next morning's run catches up (results fetching is resumable). A red X in
the Actions tab is how you'd notice — click it to see why, or send Claude a
screenshot.

## Rules of the house
- Never upload `cfbd_key.txt` (your CFBD key) — the kit's settings already
  exclude it. The daily run doesn't need it.
- The desktop CFB_Report.exe and the website are independent copies. The
  website keeps its own ledger; your desktop keeps its own.
