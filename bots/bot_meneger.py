import sys; sys.path.append(".")
import asyncio
from zoneinfo import ZoneInfo  # Python 3.9+
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
from bots.pattern_detector_bot import BotPatternDetector


class BotManager:
    def __init__(self, tz: str = "Asia/Tel_Aviv"):
        self.tz = ZoneInfo(tz)
        self.scheduler = AsyncIOScheduler(timezone=self.tz)
        self.pattern_bot = BotPatternDetector()

    async def _run_pattern_bot(self):
        try:
            # if your bot method is async:
            await self.pattern_bot.check_stocks_patterns()
        except Exception as e:
            # TODO: replace with proper logging
            print(f"[pattern_bot] error: {e!r}")

    def schedule_jobs(self):
        # Run every day at 08:00 local tz
        self.scheduler.add_job(
            self._run_pattern_bot,
            CronTrigger(hour=8, minute=0),
            id="pattern_daily",
            max_instances=1,          # avoid overlapping runs
            misfire_grace_time=900,   # 15 min grace if app was asleep
            coalesce=True             # collapse missed runs into one
        )

    def _fmt_tdelta(self, td: timedelta) -> str:
        secs = int(max(td.total_seconds(), 0))
        h, r = divmod(secs, 3600)
        m, s = divmod(r, 60)
        if h: return f"{h}h {m}m {s}s"
        if m: return f"{m}m {s}s"
        return f"{s}s"

    def start(self):
        self.schedule_jobs()
        self.scheduler.start()

        # print next run for the pattern bot
        job = self.scheduler.get_job("pattern_daily")
        if job and job.next_run_time:
            nxt = job.next_run_time.astimezone(self.tz)
            now = datetime.now(self.tz)
            print(f"[pattern_bot] next run at {nxt:%Y-%m-%d %H:%M %Z} "
                  f"(in {self._fmt_tdelta(nxt - now)})")

    async def run_forever(self):
        """Only needed if you don't already have a Discord client loop running."""
        self.start()
        while True:
            await asyncio.sleep(3600)


if __name__ == "__main__":
    # If you don't have another event loop (like discord.Client()),
    # keep this script running:
    asyncio.run(BotManager().run_forever())