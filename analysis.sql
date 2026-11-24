-- EXP-PODVID-014 user-level intent-to-treat metrics, 14-day window.
-- Run with daily_user_activity and users_experiment_map loaded into SQLite.
WITH ranked_activity AS (
    SELECT a.*,
           ROW_NUMBER() OVER (
               PARTITION BY Date, User_ID, Podcast_Minutes_Listened,
                            Music_Minutes_Listened, Discovery_Feed_Clicks, App_Crashes
               ORDER BY rowid
           ) AS exact_row_number
    FROM daily_user_activity AS a
),
deduplicated_activity AS (
    SELECT Date, User_ID, Podcast_Minutes_Listened,
           Music_Minutes_Listened, Discovery_Feed_Clicks, App_Crashes
    FROM ranked_activity
    WHERE exact_row_number = 1
),
valid_activity AS (
    SELECT a.*
    FROM deduplicated_activity AS a
    INNER JOIN users_experiment_map AS u ON u.User_ID = a.User_ID
    WHERE a.Podcast_Minutes_Listened IS NULL
       OR a.Podcast_Minutes_Listened <= 1440
),
user_activity AS (
    SELECT User_ID,
           SUM(Podcast_Minutes_Listened) AS podcast_total,
           SUM(Music_Minutes_Listened) AS music_total,
           SUM(Discovery_Feed_Clicks) AS clicks_total,
           SUM(App_Crashes) AS crashes_total,
           COUNT(DISTINCT Date) AS active_days,
           SUM(CASE WHEN Podcast_Minutes_Listened > 0 THEN 1 ELSE 0 END) AS podcast_days
    FROM valid_activity
    GROUP BY User_ID
),
clean_users AS (
    SELECT User_ID, Variant,
           CASE UPPER(TRIM(Region))
             WHEN 'NORTH AMERICA' THEN 'North America'
             WHEN 'EUROPE' THEN 'Europe'
             WHEN 'LATAM' THEN 'LATAM'
             WHEN 'APAC' THEN 'APAC'
             WHEN 'MEA' THEN 'MEA'
           END AS Region,
           Premium_Status, Account_Age_Days
    FROM users_experiment_map
)
SELECT u.User_ID, u.Variant, u.Region, u.Premium_Status, u.Account_Age_Days,
       COALESCE(a.podcast_total, 0) / 14.0 AS avg_daily_podcast_min,
       COALESCE(a.music_total, 0) / 14.0 AS avg_daily_music_min,
       COALESCE(a.clicks_total, 0) / 14.0 AS avg_daily_clicks,
       COALESCE(a.crashes_total, 0) / 14.0 AS avg_daily_crashes,
       COALESCE(a.active_days, 0) / 14.0 AS active_day_rate,
       COALESCE(a.podcast_days, 0) AS podcast_days,
       CASE WHEN COALESCE(a.podcast_total, 0) > 0 THEN 1 ELSE 0 END AS podcast_listener,
       CASE WHEN COALESCE(a.podcast_total, 0) + COALESCE(a.music_total, 0) > 0
            THEN COALESCE(a.podcast_total, 0) * 1.0 /
                 (COALESCE(a.podcast_total, 0) + COALESCE(a.music_total, 0))
            ELSE 0 END AS podcast_listening_share
FROM clean_users AS u
LEFT JOIN user_activity AS a ON a.User_ID = u.User_ID
ORDER BY u.User_ID;
