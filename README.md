# Adobe Stock Similar Content Checker

A beginner-friendly SaaS-style Streamlit MVP for Adobe Stock contributors. The app helps users upload image batches, detect visually similar content, save scan history, and test subscription limits with a local SQLite database.

Important disclaimer: this app is only a helper. It can reduce the chance of missing obvious similar images, but it does not guarantee Adobe Stock approval and it is not Adobe's official review process.

## Current Features

- User sign up, login, and logout.
- Public landing page for beta visitors.
- Privacy and trust notices on the upload workflow.
- Admin-only beta panel using the `ADMIN_EMAIL` environment variable.
- SQLite local database.
- Password hashing with `bcrypt`.
- User-specific scans and reports separated by `user_id`.
- Dashboard with current plan, monthly usage, image limit, and total scan count.
- Subscription plans loaded from SQLite.
- Manual beta payment requests for paid plan upgrades.
- Project/client folder names.
- Scan history with saved CSV report data.
- Multiple JPG, JPEG, and PNG uploads.
- Pretrained ResNet50 embedding extraction with PyTorch and torchvision.
- Proper preprocessing: resize, center crop, tensor conversion, and ImageNet normalization.
- Cosine similarity comparison with scikit-learn.
- Risky pairs, near duplicates, similar groups, CSV export, keep/remove workflow, and cleaned batch ZIP export where the plan allows it.
- Upload Readiness Report for Starter, Pro, and Agency users.
- Best Shot Selector for similar groups.
- OpenCV quality scoring for resolution, sharpness, brightness, and blur/exposure warnings.

## Subscription Plans

Plan prices, limits, and feature access are stored in the SQLite `subscription_plans` table. The values below are only the default seed values created when the database is first initialized or reset.

| Plan | Price | Scans / Month | Images / Scan | CSV Export | Cleaned ZIP Export | Readiness Report | Batch History | Client Folders |
|---|---:|---:|---:|---|---|---|---|---|
| Free | $0/month | 3 | 20 | Yes | No | No | No | No |
| Starter | $5/month | 30 | 100 | Yes | Yes | Yes | No | No |
| Pro | $12/month | 150 | 300 | Yes | Yes | Yes | Yes | No |
| Agency | $29/month | 500 | 500 | Yes | Yes | Yes | Yes | Yes |

Payment gateway integration is not connected yet. The Subscription page creates a pending manual payment request for paid plans. An admin must approve the request in `Admin Panel > Finance` before the user's plan changes. Admin users can also edit plan prices, monthly limits, image limits, and feature access from the Admin Panel.

## Manual Payment Workflow

This MVP uses a local SQLite `payments` table for beta billing.

1. A user opens `Subscription`.
2. The user selects a manual payment method such as `Bank Transfer`, `PayHere Manual`, or `PayPal/Wise Manual`.
3. The user clicks a paid plan upgrade button.
4. The app creates a pending payment reference such as `PAY-000001`.
5. The user's plan does not change yet.
6. The admin opens `Admin Panel > Finance`.
7. The admin reviews the pending payment request.
8. If the payment is valid, the admin clicks `Approve and Activate Plan`.
9. The payment status becomes `paid` and the user's plan is upgraded.
10. If the payment is not valid, the admin clicks `Reject Payment`.

Future gateways such as PayHere, Paddle, Lemon Squeezy, or Stripe can use the existing payment fields:

- `gateway_name`
- `gateway_payment_id`
- `gateway_customer_id`
- `gateway_checkout_url`
- `webhook_payload`

Do not store credit card numbers or bank account secrets in this app.

## Lemon Squeezy Readiness

The app is prepared for Lemon Squeezy, but the current beta workflow still supports manual payments.

1. Create products/plans in Lemon Squeezy for Starter, Pro, and Agency.
2. Copy each plan's Lemon Squeezy variant ID.
3. Add Lemon Squeezy secrets to `.env`.
4. Log in as admin.
5. Open `Admin Panel > Plans`.
6. Add the Lemon Squeezy variant ID for each paid plan.
7. Add a fallback Lemon Squeezy checkout URL for each paid plan.
8. Save the plan.
9. On the user `Subscription` page, plans with a checkout URL show `Pay with Lemon Squeezy`.
10. Plans without a checkout URL continue using the manual payment request flow.

When the user clicks `Pay with Lemon Squeezy`, the app sends them to the checkout URL. It does not auto-upgrade the user immediately. The plan should activate only after successful payment confirmation.

For production, add webhook processing later using a small backend such as FastAPI:

- Lemon Squeezy sends webhook event to FastAPI.
- FastAPI verifies `LEMON_SQUEEZY_WEBHOOK_SECRET`.
- FastAPI updates the SQLite/PostgreSQL `payments` row.
- FastAPI marks the payment as `paid`.
- FastAPI updates the user's plan.

Do not implement webhook handling directly inside Streamlit for production billing.

## Files in This Project

- `app.py`: Main Streamlit SaaS app and image scanner.
- `database.py`: SQLite database, authentication, users, plans, projects, and scan history.
- `requirements.txt`: Python packages needed to run the app.
- `README.md`: Setup and testing instructions.
- `adobe_saas.db`: Created automatically the first time the app runs.

## Windows Setup Commands

Open PowerShell in this project folder:

```powershell
cd D:\adobe
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this command once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the Streamlit app:

```powershell
streamlit run app.py
```

Open this URL in your browser:

```text
http://localhost:8501
```

## How to Create the Database

You do not need to manually create the database.

The app calls `init_db()` when it starts. This automatically creates:

- `users`
- `projects`
- `scans`
- `subscription_plans`
- `payments`

The database file will appear here:

```text
D:\adobe\adobe_saas.db
```

## How to Sign Up and Test

1. Run the app:

```powershell
streamlit run app.py
```

2. Open:

```text
http://localhost:8501
```

3. Click the `Sign Up` tab.
4. Enter your name, email, and password.
5. Your new account starts on the Free plan.
6. Go to `New Scan`.
7. Enter a project/client folder name.
8. Enter a batch name.
9. Upload images.
10. Click `Process Scan`.

## Beta Privacy Notes

- Uploaded images are used only for similarity analysis.
- Uploaded images are not used for AI training.
- In this local MVP, uploaded image bytes are held temporarily in Streamlit session memory during the scan so previews, CSV reports, and cleaned ZIP export can work.
- The app does not write uploaded image files permanently to disk.
- Scan history stores report metadata and CSV rows, not the original uploaded image files.
- `cleanup_temporary_scan_files()` is included as a beta cleanup hook. It is currently a no-op because the app does not create temporary image files on disk.
- StockGuard AI is not affiliated with Adobe and does not guarantee Adobe Stock approval.

## Environment Variables

Set these before beta deployment:

```powershell
$env:ADMIN_EMAIL="your-admin-email@example.com"
$env:FEEDBACK_EMAIL="support@example.com"
```

Optional reserved variable for future production security work:

```powershell
$env:APP_SECRET_KEY="replace-with-a-long-random-secret"
```

Optional local `.env` setup:

1. Copy `.env.example` to `.env`.
2. Change `ADMIN_EMAIL` and `APP_SECRET_KEY`.
3. Do not commit `.env`.

Supported variables:

- `ADMIN_EMAIL`: account email that should become admin.
- `APP_SECRET_KEY`: reserved production secret value.
- `DATABASE_PATH`: SQLite database path, default `adobe_saas.db`.
- `MAX_IMAGE_SIZE_MB`: max upload size per image, default `15`.
- `APP_ENV`: `development` or production environment label.
- `FEEDBACK_EMAIL`: beta feedback contact.
- `LEMON_SQUEEZY_API_KEY`: Lemon Squeezy API key for future gateway work.
- `LEMON_SQUEEZY_STORE_ID`: Lemon Squeezy store ID.
- `LEMON_SQUEEZY_STARTER_VARIANT_ID`: Starter plan variant ID.
- `LEMON_SQUEEZY_PRO_VARIANT_ID`: Pro plan variant ID.
- `LEMON_SQUEEZY_AGENCY_VARIANT_ID`: Agency plan variant ID.
- `LEMON_SQUEEZY_WEBHOOK_SECRET`: webhook signing secret for future FastAPI webhook validation.

`APP_SECRET_KEY` is documented for deployment readiness, but the current Streamlit MVP does not require it for login because authentication is handled with bcrypt password hashes and SQLite sessions.

`FEEDBACK_EMAIL` controls the beta feedback contact shown on the public landing page.

## Security Checklist

- Use `.env` for secrets and never commit it.
- Use a strong admin email/password.
- Change `APP_SECRET_KEY` before production deployment.
- Set `MAX_IMAGE_SIZE_MB` to a safe upload size for your server.
- Use HTTPS when deployed.
- Keep dependencies updated.
- Uploaded images are validated server-side by extension, size, Pillow verification, and actual JPEG/PNG format.
- Passwords are stored as bcrypt hashes only, never plain text.
- Admin Panel access is guarded by admin role / `ADMIN_EMAIL` checks.
- CSV exports neutralize spreadsheet formula injection prefixes.
- Clean ZIP export uses sanitized filenames only.
- Uploaded files are not stored permanently in this beta unless you add storage later.
- For production scale, move from SQLite/local memory to PostgreSQL plus object storage such as S3/R2.

## How to Test the Admin Panel

1. Set `ADMIN_EMAIL` to the email address of your admin account.
2. Start the app:

```powershell
streamlit run app.py
```

3. Sign up or log in with that same email.
4. The sidebar will show `Admin Panel`.
5. Open it to:
   - View all users
   - See each user's current plan
   - See monthly scans used
   - See total scan count
   - Change a user plan manually
   - Review pending manual payment requests
   - Approve paid upgrades
   - Reject invalid payment requests
   - View revenue metrics
   - Disable or enable a user account
   - Edit subscription plan prices, limits, and feature access

If a user is disabled, login for that account is blocked.

## How to Test Manual Payments

1. Log in as a normal user.
2. Open `Subscription`.
3. Choose a paid plan such as Starter or Pro.
4. Add an optional payment note.
5. Click the upgrade button.
6. Confirm that a pending payment reference appears.
7. Log out.
8. Log in as the admin account configured by `ADMIN_EMAIL`.
9. Open `Admin Panel > Finance`.
10. Find the pending payment in the compact payment table.
11. Open the manage expander.
12. Click `Approve and Activate Plan`.
13. Log back in as the normal user and confirm the plan changed.

Free plan selection still changes immediately because it does not require payment approval.

## How to Test Database-Driven Plan Settings

1. Log in as the admin account.
2. Open `Admin Panel`.
3. Expand `Edit Starter plan`.
4. Change the price, monthly scan limit, or images per scan limit.
5. Click `Save plan settings`.
6. Open `Subscription` and confirm the updated values appear.
7. Open `New Scan` and confirm upload limits use the updated database value.

## How to Reset the Database

Stop Streamlit first. Then delete the local SQLite file:

```powershell
Remove-Item .\adobe_saas.db
```

Restart the app:

```powershell
streamlit run app.py
```

The app will recreate the database tables automatically.

Warning: this removes all local users, projects, and scan history.

## How to Test Free Plan Restrictions

With the default database seed values, the Free plan allows:

- 3 scans per month
- 20 images per scan
- CSV export only

Test image limit:

1. Create a Free account.
2. Go to `New Scan`.
3. Upload more than 20 images.
4. The app should block the scan and show an upgrade message.

Test monthly scan limit:

1. Create a Free account.
2. Complete 3 scans with 20 or fewer images each.
3. Try a 4th scan.
4. The app should block the scan and show an upgrade message.

Test CSV export:

1. Complete a scan that has risky pairs.
2. Click `Download CSV report`.

Test cleaned ZIP restriction:

1. Complete a scan on the Free plan.
2. The app should show that cleaned batch ZIP export is available on Starter, Pro, and Agency plans.

## How to Simulate Upgrade to Pro

1. Log in.
2. Open the sidebar.
3. Click `Subscription`.
4. Click `Upgrade to Pro`.
5. The dashboard should now show the Pro plan.

With the default database seed values, the Pro plan allows:

- 150 scans per month
- 300 images per scan
- CSV export
- Cleaned batch ZIP export
- Batch history

## Deployment Notes: Render or Railway

This Streamlit MVP can be deployed on Render or Railway as a web service.

Recommended start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Required setup:

- Add all packages from `requirements.txt`.
- Set `ADMIN_EMAIL` in the platform environment variables.
- Set `FEEDBACK_EMAIL` to your beta support email.
- Use persistent storage if you want SQLite data to survive restarts.
- For serious production use, move from SQLite to a managed database such as PostgreSQL.
- For uploaded files at scale, use cloud object storage instead of session memory.

Render high-level steps:

1. Push this project to GitHub.
2. Create a new Render Web Service.
3. Select the repo.
4. Install command:

```bash
pip install -r requirements.txt
```

5. Start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

6. Add environment variable:

```text
ADMIN_EMAIL=your-admin-email@example.com
```

Railway high-level steps:

1. Create a Railway project from GitHub.
2. Add the same install/start commands.
3. Add `ADMIN_EMAIL` in Variables.
4. Add persistent storage or switch to PostgreSQL for production beta data.

## How to View Scan History

1. Log in.
2. Click `Scan History`.
3. Select a previous scan.
4. Review scan summary metrics.
5. Download the saved CSV report if risky pairs exist.

Only the logged-in user can see their own scans because scan queries filter by `user_id`.

## Risk Labels

- 80% to 89.99%: Possible Similar Content
- 90% to 94.99%: High Risk
- 95% and above: Very High Risk / Near Duplicate

## Common Errors and Fixes

### `python` is not recognized

Python is not installed or not added to PATH.

Fix:

- Install Python from https://www.python.org/downloads/
- During installation, check "Add Python to PATH".
- Close and reopen PowerShell.

### PowerShell will not activate `.venv`

You may see an execution policy error.

Fix:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### `No module named bcrypt`

The new SaaS version needs bcrypt.

Fix:

```powershell
pip install -r requirements.txt
```

### `No module named cv2`

The Upload Readiness Report needs OpenCV.

Fix:

```powershell
pip install -r requirements.txt
```

## Upload Readiness Report

Available on Starter, Pro, and Agency plans.

The report calculates:

- Resolution score
- Sharpness score using OpenCV Laplacian variance
- Brightness score
- Possible blur warning
- Too dark / too bright warning
- Overall quality score out of 100
- Similarity risk
- Upload readiness status

Statuses:

- Ready to Upload
- Review Needed
- Remove Recommended

Free users see a paid feature preview card and can upgrade from inside the app.

### The first app run is slow

The app downloads pretrained ResNet50 weights the first time it runs.

Fix:

- Wait for the first download to finish.
- Later runs should be faster because the model is cached.

### CUDA is not being used

The app uses CUDA only when your computer has a compatible NVIDIA GPU and a compatible PyTorch install. CPU mode is normal and works fine for small batches.

Fix:

- Use smaller image batches if CPU processing is slow.
- For GPU setup, follow the official PyTorch install selector: https://pytorch.org/get-started/locally/

### Corrupted image warning

The app skips files that Pillow cannot open as images.

Fix:

- Re-export the image as JPG or PNG.
- Make sure the file is not damaged.
