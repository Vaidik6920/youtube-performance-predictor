import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier, Pool
from textblob import TextBlob
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG & STYLES
# ============================================================

st.set_page_config(
    page_title="YouTube Performance Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #FF0000;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 2px solid #FF0000;
        padding-bottom: 10px;
    }
    .sector-badge {
        display:inline-block;
        padding:4px 8px;
        border-radius:6px;
        margin:2px 4px 2px 0;
        font-size:12px;
        background-color:#f1f3f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODELS & ENCODERS
# ============================================================

@st.cache_resource(show_spinner=True)
def load_all_models():
    # engagement + regressors
    eng_model = joblib.load("model_engagement_rf_sentiment.pkl")
    le_eng = joblib.load("label_encoder_engagement_tier.pkl")
    scaler_eng = joblib.load("scaler_engagement_sentiment.pkl")

    views_model = joblib.load("model_views_regressor_sentiment.pkl")
    likes_model = joblib.load("model_likes_regressor_sentiment.pkl")
    comments_model = joblib.load("model_comments_regressor_sentiment.pkl")
    eng_rate_model = joblib.load("model_engagement_rate_regressor_sentiment.pkl")

    # sector model
    sector_model = CatBoostClassifier()
    sector_model.load_model("sector_model.cbm")
    le_sector = joblib.load("label_encoder_sector.pkl")
    feature_cols_sector = joblib.load("sector_feature_cols.joblib")
    text_idx = joblib.load("sector_text_idx.joblib")
    cat_idx = joblib.load("sector_cat_idx.joblib")
    cat_encoders = joblib.load("sector_cat_encoders.joblib")

    # tier models
    tier_feature_cols = joblib.load("tier_metric_feature_cols.pkl")
    tier_model_vps = CatBoostClassifier()
    tier_model_vps.load_model("cb_tier_views_per_sub.cbm")
    tier_model_lpv = CatBoostClassifier()
    tier_model_lpv.load_model("cb_tier_likes_per_view.cbm")
    tier_model_er = CatBoostClassifier()
    tier_model_er.load_model("cb_tier_eng_rate.cbm")

    # hashtag models
    tfidf_hashtags = joblib.load("tfidf_hashtags.pkl")
    mlb_hashtags = joblib.load("mlb_hashtags.pkl")
    hashtag_model = joblib.load("hashtag_model_ovr_lr.pkl")

    return {
        "eng_model": eng_model,
        "views_model": views_model,
        "likes_model": likes_model,
        "comments_model": comments_model,
        "eng_rate_model": eng_rate_model,
        "le_eng": le_eng,
        "scaler_eng": scaler_eng,
        "sector_model": sector_model,
        "le_sector": le_sector,
        "sector_feature_cols": feature_cols_sector,
        "sector_text_idx": text_idx,
        "sector_cat_idx": cat_idx,
        "sector_cat_encoders": cat_encoders,
        # tier models
        "tier_feature_cols": tier_feature_cols,
        "tier_model_vps": tier_model_vps,
        "tier_model_lpv": tier_model_lpv,
        "tier_model_er": tier_model_er,
        # hashtags
        "tfidf_hashtags": tfidf_hashtags,
        "mlb_hashtags": mlb_hashtags,
        "hashtag_model": hashtag_model,
    }

models = load_all_models()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_sentiment(text: str):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def analyze_sentiment_comprehensive(title: str, description: str):
    t_pol, t_sub = extract_sentiment(title)
    d_pol, d_sub = extract_sentiment(description)
    c_pol, c_sub = extract_sentiment(f"{title} {description}")

    def label(p):
        if p > 0.3:
            return "Positive"
        elif p < -0.3:
            return "Negative"
        return "Neutral"

    return {
        "title": {"polarity": t_pol, "subjectivity": t_sub, "category": label(t_pol)},
        "description": {"polarity": d_pol, "subjectivity": d_sub, "category": label(d_pol)},
        "combined": {"polarity": c_pol, "subjectivity": c_sub, "category": label(c_pol)},
    }

def build_engagement_features(title, description, duration, followers):
    title_str = str(title).strip()
    desc_str = str(description).strip()

    title_len = len(title_str)
    desc_len = len(desc_str)
    title_words = len(title_str.split())
    desc_words = len(desc_str.split())

    t_pol, t_sub = extract_sentiment(title_str)
    d_pol, d_sub = extract_sentiment(desc_str)
    c_pol, c_sub = extract_sentiment(f"{title_str} {desc_str}")

    feats = np.array([[
        title_len,
        desc_len,
        title_words,
        desc_words,
        t_pol,
        t_sub,
        d_pol,
        d_sub,
        c_pol,
        c_sub,
        float(duration),
        float(followers),
    ]], dtype=float)

    return models["scaler_eng"].transform(feats)

def build_sector_pool_from_input(
    title,
    description,
    fine_category_text="",
    youtube_categories_text="",
    duration_seconds=0.0,
    followers=0.0,
    audience_engagement_index=0.0,
    views_per_subscriber=0.0,
    likes_per_subscriber=0.0,
    comments_per_subscriber=0.0,
    engagement_rate=0.0,
    like_rate=0.0,
    comment_rate=0.0,
    like_to_comment_ratio=0.0,
    video_completeness_score=0.8,
    tts_quality_indicator=0.7,
    production_polish_score=0.7,
):
    feature_cols = models["sector_feature_cols"]

    title = str(title).strip()
    description = str(description).strip()
    fine_category_text = str(fine_category_text).strip()
    youtube_categories_text = str(youtube_categories_text).strip()

    all_text = f"{title} {description} {fine_category_text} {youtube_categories_text}".strip()

    row = {col: 0 for col in feature_cols}
    if "all_text" in row:
        row["all_text"] = all_text

    title_len = len(title)
    desc_len = len(description)
    title_wc = len(title.split()) if title else 0
    desc_wc = len(description.split()) if description else 0
    t_pol, t_sub = extract_sentiment(title)
    d_pol, d_sub = extract_sentiment(description)
    c_pol, c_sub = extract_sentiment(title + " " + description)

    def set_if_exists(col, val):
        if col in row:
            row[col] = float(val)

    set_if_exists("title_length_chars", title_len)
    set_if_exists("title_word_count", title_wc)
    set_if_exists("description_length_chars", desc_len)
    set_if_exists("description_word_count", desc_wc)
    set_if_exists("title_sentiment", t_pol)
    set_if_exists("title_subjectivity", t_sub)
    set_if_exists("description_sentiment", d_pol)
    set_if_exists("description_subjectivity", d_sub)
    set_if_exists("meta_description_sentiment", c_pol)
    set_if_exists("duration_seconds", duration_seconds)
    set_if_exists("youtube_channel_follower_count", followers)
    set_if_exists("audience_engagement_index", audience_engagement_index)
    set_if_exists("video_completeness_score", video_completeness_score)
    set_if_exists("tts_quality_indicator", tts_quality_indicator)
    set_if_exists("production_polish_score", production_polish_score)
    set_if_exists("views_per_subscriber", views_per_subscriber)
    set_if_exists("likes_per_subscriber", likes_per_subscriber)
    set_if_exists("comments_per_subscriber", comments_per_subscriber)
    set_if_exists("engagement_rate", engagement_rate)
    set_if_exists("like_rate", like_rate)
    set_if_exists("comment_rate", comment_rate)
    set_if_exists("like_to_comment_ratio", like_to_comment_ratio)

    X = pd.DataFrame([row])[feature_cols]
    pool = Pool(X, text_features=models["sector_text_idx"])
    return pool

# extra helpers for tier models
CALL_TO_ACTION_WORDS = [
    "subscribe", "sub", "like", "comment", "share",
    "watch", "click", "link", "join", "follow",
]

def readability_features(text: str):
    words = text.split()
    n_words = len(words)
    avg_word_len = (sum(len(w) for w in words) / n_words) if n_words > 0 else 0.0
    n_sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
    avg_sentence_len = n_words / n_sentences
    flesch_proxy = 206.835 - 1.015 * avg_sentence_len - 0.84 * avg_word_len
    return avg_word_len, avg_sentence_len, flesch_proxy

def keyword_features(text: str):
    text_l = text.lower()
    count = sum(text_l.count(w) for w in CALL_TO_ACTION_WORDS)
    present = int(count > 0)
    return count, present

def build_tier_features(title, description, duration, followers):
    title = str(title)
    description = str(description)
    full_text = f"{title} {description}"

    t_pol, t_sub = extract_sentiment(title)
    d_pol, d_sub = extract_sentiment(description)
    c_pol, c_sub = extract_sentiment(full_text)

    avg_word_len, avg_sentence_len, flesch_proxy = readability_features(full_text)
    cta_count, cta_present = keyword_features(full_text)

    row = {
        "duration_seconds": float(duration),
        "subs": float(followers),
        "title_length_chars": len(title),
        "description_length_chars": len(description),
        "title_word_count": len(title.split()),
        "description_word_count": len(description.split()),
        "title_sentiment": t_pol,
        "title_subjectivity": t_sub,
        "description_sentiment": d_pol,
        "description_subjectivity": d_sub,
        "combined_sentiment": c_pol,
        "combined_subjectivity": c_sub,
        "avg_word_length": avg_word_len,
        "avg_sentence_length": avg_sentence_len,
        "flesch_proxy": flesch_proxy,
        "cta_word_count": cta_count,
        "cta_present": cta_present,
    }

    cols = models["tier_feature_cols"]
    X = pd.DataFrame([row])[cols]
    return X

def predict_all(title, description, duration, followers,
                fine_category_text="", youtube_categories_text=""):
    eng_feats = build_engagement_features(title, description, duration, followers)

    eng_pred_int = models["eng_model"].predict(eng_feats)[0]
    eng_proba = models["eng_model"].predict_proba(eng_feats)[0]
    eng_tier = models["le_eng"].inverse_transform([eng_pred_int])[0]

    log_views = models["views_model"].predict(eng_feats)[0]
    log_likes = models["likes_model"].predict(eng_feats)[0]
    log_comments = models["comments_model"].predict(eng_feats)[0]
    eng_rate = models["eng_rate_model"].predict(eng_feats)[0]

    views = max(1, int(np.expm1(log_views)))
    likes = max(1, int(np.expm1(log_likes)))
    comments = max(0, int(np.expm1(log_comments)))
    eng_rate = float(np.clip(eng_rate, 0.0, 1.0))

    vps = views / max(1.0, followers)
    lps = likes / max(1.0, followers)
    cps = comments / max(1.0, followers)
    like_to_comment = likes / (comments + 1e-6) if comments > 0 else 0.0

    sector_pool = build_sector_pool_from_input(
        title=title,
        description=description,
        fine_category_text=fine_category_text,
        youtube_categories_text=youtube_categories_text,
        duration_seconds=duration,
        followers=followers,
        audience_engagement_index=0.0,
        views_per_subscriber=vps,
        likes_per_subscriber=lps,
        comments_per_subscriber=cps,
        engagement_rate=eng_rate,
        like_rate=likes / max(1.0, views),
        comment_rate=comments / max(1.0, views),
        like_to_comment_ratio=like_to_comment,
    )

    sector_proba = models["sector_model"].predict_proba(sector_pool)[0]
    sector_idx = int(np.argmax(sector_proba))
    sector = models["le_sector"].inverse_transform([sector_idx])[0]

    return {
        "eng_tier": eng_tier,
        "eng_proba": eng_proba,
        "views": views,
        "likes": likes,
        "comments": comments,
        "eng_rate": eng_rate,
        "sector": sector,
        "sector_proba": sector_proba,
    }

# ============================================================
# HEADER
# ============================================================

st.markdown(
    "<h1 style='text-align:center;color:#FF0000;'>YOUTUBE PERFORMANCE PREDICTOR</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <p style='text-align:center;color:gray;max-width:900px;margin:auto;'>
    This tool uses machine learning models trained on thousands of YouTube videos.
    It analyses your title, description, basic channel stats and historical engagement patterns
    to estimate engagement tier, likely sector, and approximate views/engagement.
    Predictions are probabilistic guidance, not exact guarantees.
    </p>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("VIDEO DETAILS")

title = st.sidebar.text_input(
    "VIDEO TITLE (required)",
    placeholder="e.g., Premier League match highlights and goals",
    max_chars=100,
)

description = st.sidebar.text_area(
    "VIDEO DESCRIPTION (required)",
    placeholder="Describe your video here...",
    height=120,
    max_chars=1000,
)

st.sidebar.markdown("---")
st.sidebar.subheader("CONTENT CATEGORIES")

FINE_CATEGORY_OPTIONS = [
    "",
    # Education
    "Online Courses & Tutorials",
    "Programming Tutorials",
    "Math & Science Lessons",
    "History & Geography",
    "Language Learning",
    "Career & Skill Development",
    # News & Politics
    "Political Commentary",
    "Elections & Campaigns",
    "Policy & Government Analysis",
    "International Affairs",
    "Business & Economy News",
    # Sports
    "Match Highlights",
    "Esports & Gaming Tournaments",
    "Match Analysis & Tactics",
    # Lifestyle
    "Fashion & Style",
    "Beauty & Skincare",
    "Travel Vlogs",
    "Home & Interior",
    "Food & Cooking",
    "Productivity & Self‑Improvement",
    "Relationships & Personal Life",
]

fine_category_text = st.sidebar.selectbox(
    "FINE CATEGORY (optional)",
    options=FINE_CATEGORY_OPTIONS,
    index=0,
    help="Choose the closest fine-grained topic for this video.",
)

st.sidebar.markdown("---")

YOUTUBE_CATEGORY_OPTIONS = [
    "",
    "Education",
    "Entertainment",
    "Science & Technology",
    "News & Politics",
    "Sports",
    "Lifestyle",
    "Autos & Vehicles",
]

youtube_categories_text = st.sidebar.selectbox(
    "YOUTUBE CATEGORY (optional)",
    options=YOUTUBE_CATEGORY_OPTIONS,
    index=0,
    help="Choose the main YouTube category you would publish under.",
)

st.sidebar.markdown("---")
st.sidebar.header("VIDEO STATS")

duration = st.sidebar.slider(
    "DURATION (seconds)",
    min_value=10,
    max_value=7200,
    value=600,
    step=30,
)

followers = st.sidebar.number_input(
    "CHANNEL FOLLOWERS",
    min_value=0,
    value=50000,
    step=1000,
)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("PREDICT PERFORMANCE", use_container_width=True)

# ============================================================
# MAIN
# ============================================================

if predict_btn:
    if not title.strip() or not description.strip():
        st.error("Please enter both title and description.")
    else:
        results = predict_all(
            title, description, duration, followers,
            fine_category_text=fine_category_text,
            youtube_categories_text=youtube_categories_text,
        )
        sentiment = analyze_sentiment_comprehensive(title, description)

        # blended views: subscriber baseline + model
        baseline_views = 0.1 * followers  # tune factor as needed
        model_views = results["views"]
        final_views = int(0.8 * baseline_views + 0.2 * model_views)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["ENGAGEMENT", "SECTOR", "METRICS", "SENTIMENT", "HASHTAGS"]
        )

        # ---------------- ENGAGEMENT ----------------
        with tab1:
            st.markdown(
                "<div class='section-header'>ENGAGEMENT TIER PREDICTION</div>",
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)

            with c1:
                tier = results["eng_tier"]
                tier_prob = results["eng_proba"][list(models["le_eng"].classes_).index(tier)] * 100
                if tier == "HIGH":
                    color = "#28a745"
                    emoji = "🔥"
                elif tier == "MID":
                    color = "#ffc107"
                    emoji = "⚡"
                else:
                    color = "#dc3545"
                    emoji = "⬇️"

                st.markdown(
                    f"""
                    <div style='background:{color};color:white;padding:20px;border-radius:10px;text-align:center;'>
                        <div style='font-size:40px;'>{emoji}</div>
                        <div style='font-size:28px;font-weight:bold;'>{tier}</div>
                        <div style='font-size:14px;margin-top:10px;'>{tier_prob:.1f}% confidence</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                low_prob = results["eng_proba"][list(models["le_eng"].classes_).index("LOW")] * 100
                mid_prob = results["eng_proba"][list(models["le_eng"].classes_).index("MID")] * 100
                st.metric("LOW Probability", f"{low_prob:.1f}%")
                st.metric("MID Probability", f"{mid_prob:.1f}%")

            with c3:
                st.markdown("**Tier Meaning:**")
                st.write("- 🔥 HIGH: Strong engagement expected")
                st.write("- ⚡ MID: Moderate engagement")
                st.write("- ⬇️ LOW: Weak engagement")

            st.markdown("### Tier probabilities")
            tier_labels = ["LOW", "MID", "HIGH"]
            tier_vals = [
                results["eng_proba"][list(models["le_eng"].classes_).index("LOW")] * 100,
                results["eng_proba"][list(models["le_eng"].classes_).index("MID")] * 100,
                results["eng_proba"][list(models["le_eng"].classes_).index("HIGH")] * 100,
            ]
            fig_tier = go.Figure(
                data=[
                    go.Bar(
                        x=tier_labels,
                        y=tier_vals,
                        marker=dict(color=["#dc3545", "#ffc107", "#28a745"]),
                        text=[f"{v:.1f}%" for v in tier_vals],
                        textposition="auto",
                    )
                ]
            )
            fig_tier.update_layout(
                xaxis_title="Tier",
                yaxis_title="Probability (%)",
                height=280,
                showlegend=False,
            )
            st.plotly_chart(fig_tier, use_container_width=True)

            st.caption(
                "Engagement tier model was evaluated on historical videos; it correctly identified "
                "the tier in roughly 60–70% of cases, with best performance on clearly high or low engagement."
            )

        # ---------------- SECTOR ----------------
        with tab2:
            st.markdown(
                "<div class='section-header'>SECTOR Suggestion</div>",
                unsafe_allow_html=True,
            )
            sector = results["sector"]
            sector_proba = results["sector_proba"]
            sector_names = list(models["le_sector"].classes_)
            sector_probs_pct = np.array(sector_proba) * 100

            top_idx = int(np.argmax(sector_probs_pct))
            top_sector = sector_names[top_idx]
            top_prob = sector_probs_pct[top_idx]

            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(
                    f"""
                    <div style='background:#007bff;color:white;padding:20px;border-radius:10px;text-align:center;'>
                        <div style='font-size:24px;font-weight:bold;'>{top_sector}</div>
                        <div style='font-size:14px;margin-top:10px;'>{top_prob:.1f}% confidence</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown("**Sector Suggestions:**")
                fig_sector = go.Figure(
                    data=[
                        go.Bar(
                            x=sector_probs_pct,
                            y=sector_names,
                            orientation="h",
                            marker=dict(color=sector_probs_pct, colorscale="Blues"),
                            text=[f"{v:.1f}%" for v in sector_probs_pct],
                            textposition="auto",
                        )
                    ]
                )
                fig_sector.update_layout(
                    margin=dict(l=0, r=10, t=10, b=10),
                    xaxis_title="Probability (%)",
                    height=260,
                    showlegend=False,
                )
                st.plotly_chart(fig_sector, use_container_width=True)

            diffs = top_prob - sector_probs_pct
            close_mask = diffs < 20.0
            close_indices = np.where(close_mask)[0]
            close_indices = [i for i in close_indices if i != top_idx]

            if close_indices:
                st.markdown("### Other plausible sectors")
                for i in close_indices:
                    name = sector_names[i]
                    prob = sector_probs_pct[i]
                    st.write(
                        f"- **{name}** also has a significant probability ({prob:.1f}%). "
                        "sharing patterns with this sector in the training data."
                    )
            else:
                st.markdown("### Model confidence")
                st.write(
                    "The top sector is at least 20 percentage points higher than all others, "
                    "so the model is relatively confident about this classification."
                )

            st.info(
                "Sector model is trained on title, description, category text and numeric features from training videos. "
                "On a held‑out validation set it achieved about 0.67 accuracy and 0.66 macro‑F1 across six sectors, "
                "with per‑sector accuracy ranging from roughly 0.55 (hardest) to 0.73 (easiest)."
            )

        # ---------------- METRICS ----------------
        with tab3:
            st.markdown(
                "<div class='section-header'>PREDICTED METRICS</div>",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.metric("PREDICTED VIEWS", f"{final_views:,}")
            with c2:
                st.metric("PREDICTED ENGAGEMENT RATE", f"{results['eng_rate']:.2%}")

            st.markdown("### Model & sector statistics")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Model evaluation (held‑out data)**")
                st.write("- Sector classifier: accuracy ≈ 0.67, macro‑F1 ≈ 0.66.")
                st.write("- Engagement tier model: accuracy ≈ 0.61, macro‑F1 ≈ 0.59.")
                st.write("- Performance tiers (views/likes/engagement): accuracy ≈ 0.50, macro‑F1 ≈ 0.48.")
                st.write("- Hashtag recommender: micro‑F1 ≈ 0.64, Jaccard ≈ 0.46 (top tags).")

            with col2:
                st.markdown("**Typical engagement‑rate by sector**")
                st.write("- Education / Sci‑Tech: ~6–8% average.")
                st.write("- Entertainment / Sports: ~4–6% average.")
                st.write("- Lifestyle / News & Politics: ~5–7% average.")

            st.markdown("### How this prediction compares to sector average")

            avg_by_sector = {
                "Education": 0.07,
                "Science & Technology": 0.07,
                "Entertainment": 0.05,
                "Sports": 0.05,
                "Lifestyle": 0.06,
                "News & Politics": 0.06,
            }
            sector_avg = avg_by_sector.get(results["sector"], 0.06)
            if sector_avg > 0:
                over_ratio = results["eng_rate"] / sector_avg
                over_pct = (over_ratio - 1.0) * 100
                direction = "above" if over_pct >= 0 else "below"
                st.write(
                    f"- For **{results['sector']}**, the model expects around **{sector_avg:.2%}** "
                    f"average engagement. This prediction is approximately **{abs(over_pct):.1f}% {direction}** that level."
                )
            else:
                st.write(
                    "- Sector average engagement is not available for this class; treat the predicted rate as a rough standalone estimate."
                )

            st.caption(
                "Engagement rate comparisons are approximate and based on aggregate patterns in the training data. "
                "Real‑world performance will also depend on thumbnail, timing, larger dataset, which will be updated in BTP-2."
            )

            # --- tier models (views/likes/eng-rate) ---
            st.markdown("### Performance tiers")

            X_tier = build_tier_features(title, description, duration, followers)

            # views per subscriber tier
            proba_vps = models["tier_model_vps"].predict_proba(X_tier)[0]
            classes_vps = models["tier_model_vps"].classes_
            top_idx_vps = int(np.argmax(proba_vps))
            st.markdown(f"**Views per subscriber tier:** {classes_vps[top_idx_vps]}")

            fig_vps = go.Figure(
                data=[
                    go.Bar(
                        x=list(classes_vps),
                        y=proba_vps * 100,
                        marker=dict(color=["#dc3545", "#ffc107", "#28a745"]),
                        text=[f"{p*100:.1f}%" for p in proba_vps],
                        textposition="auto",
                    )
                ]
            )
            fig_vps.update_layout(
                title="Views per subscriber",
                yaxis_title="Probability (%)",
                height=230,
                showlegend=False,
            )
            st.plotly_chart(fig_vps, use_container_width=True)

            # likes per view tier
            proba_lpv = models["tier_model_lpv"].predict_proba(X_tier)[0]
            classes_lpv = models["tier_model_lpv"].classes_
            top_idx_lpv = int(np.argmax(proba_lpv))
            st.markdown(f"**Likes per view tier:** {classes_lpv[top_idx_lpv]}")

            fig_lpv = go.Figure(
                data=[
                    go.Bar(
                        x=list(classes_lpv),
                        y=proba_lpv * 100,
                        marker=dict(color=["#dc3545", "#ffc107", "#28a745"]),
                        text=[f"{p*100:.1f}%" for p in proba_lpv],
                        textposition="auto",
                    )
                ]
            )
            fig_lpv.update_layout(
                title="Likes per view",
                yaxis_title="Probability (%)",
                height=230,
                showlegend=False,
            )
            st.plotly_chart(fig_lpv, use_container_width=True)

            # engagement-rate tier
            proba_er = models["tier_model_er"].predict_proba(X_tier)[0]
            classes_er = models["tier_model_er"].classes_
            top_idx_er = int(np.argmax(proba_er))
            st.markdown(f"**Engagement rate tier:** {classes_er[top_idx_er]}")

            fig_er = go.Figure(
                data=[
                    go.Bar(
                        x=list(classes_er),
                        y=proba_er * 100,
                        marker=dict(color=["#dc3545", "#ffc107", "#28a745"]),
                        text=[f"{p*100:.1f}%" for p in proba_er],
                        textposition="auto",
                    )
                ]
            )
            fig_er.update_layout(
                title="Engagement rate",
                yaxis_title="Probability (%)",
                height=230,
                showlegend=False,
            )
            st.plotly_chart(fig_er, use_container_width=True)

        # ---------------- SENTIMENT ----------------
        with tab4:
            st.markdown(
                "<div class='section-header'>SENTIMENT ANALYSIS</div>",
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**TITLE SENTIMENT**")
                t = sentiment["title"]
                st.markdown(f"**Category:** {t['category']}")
                st.metric("Polarity", f"{t['polarity']:.2f}")
                st.metric("Subjectivity", f"{t['subjectivity']:.2f}")

            with c2:
                st.markdown("**DESCRIPTION SENTIMENT**")
                d = sentiment["description"]
                st.markdown(f"**Category:** {d['category']}")
                st.metric("Polarity", f"{d['polarity']:.2f}")
                st.metric("Subjectivity", f"{d['subjectivity']:.2f}")

            with c3:
                st.markdown("**COMBINED SENTIMENT**")
                c = sentiment["combined"]
                st.markdown(f"**Category:** {c['category']}")
                st.metric("Polarity", f"{c['polarity']:.2f}")
                st.metric("Subjectivity", f"{c['subjectivity']:.2f}")

            st.caption(
                "Sentiment scores are computed using TextBlob polarity (−1 to 1) and subjectivity (0 to 1), "
                "summarising how positive/negative and opinion‑driven your text appears."
            )

        # ---------------- HASHTAGS ----------------
        with tab5:
            st.markdown(
                "<div class='section-header'>HASHTAG SUGGESTIONS</div>",
                unsafe_allow_html=True,
            )

            meta_text = f"{fine_category_text} {youtube_categories_text}".strip()
            full_text = f"{title} {description} {meta_text}".strip()

            X_hash = models["tfidf_hashtags"].transform([full_text])
            proba_hash = models["hashtag_model"].predict_proba(X_hash)[0]
            classes_hash = models["mlb_hashtags"].classes_

            idx_sorted = np.argsort(proba_hash)[::-1]

            suggestions = []
            # collect up to 5 tags with prob >= 0.10
            for idx in idx_sorted:
                if proba_hash[idx] >= 0.10:
                    suggestions.append((classes_hash[idx], float(proba_hash[idx])))
                    if len(suggestions) >= 5:
                        break

            # if none above threshold, fall back to top-1
            if not suggestions:
                top_idx = idx_sorted[0]
                suggestions = [(classes_hash[top_idx], float(proba_hash[top_idx]))]

            suggestions = suggestions[:5]

            top_tags = [f"#{tag}" for tag, _ in suggestions]
            st.markdown("**Top suggested hashtags:**")
            st.write(" ".join(top_tags))

            st.markdown("### Hashtag scores")
            fig_tags = go.Figure(
                data=[
                    go.Bar(
                        x=[f"#{t}" for t, _ in suggestions],
                        y=[p * 100 for _, p in suggestions],
                        marker=dict(color="#007bff"),
                        text=[f"{p*100:.1f}%" for _, p in suggestions],
                        textposition="auto",
                    )
                ]
            )
            fig_tags.update_layout(
                xaxis_title="Hashtag",
                yaxis_title="Confidence (%)",
                height=320,
                showlegend=False,
            )
            st.plotly_chart(fig_tags, use_container_width=True)

            st.caption(
                "Hashtag predictions are based on a multi‑label text model trained on historical "
                "titles, descriptions, metadata, and existing tags. Use them as suggestions and "
                "adjust based on your video and audience."
            )

else:
    st.info("Enter video details in the sidebar and click PREDICT PERFORMANCE to see predictions.")
