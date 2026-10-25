"""Tkinter keyboard interface for inspecting and manually driving RacingEnv."""
from __future__ import annotations

import tkinter as tk
import numpy as np

from .env import (
    RacingEnv, make_track, DT, MAX_SPEED, MAX_DELTA_V, MAX_DELTA_HEADING,
    RAY_ANGLES_DEG, HALF_TRACK_WIDTH,
)

WIDTH, HEIGHT = 1180, 800
SCALE = 5.2
FRAME_MS = int(round(DT * 1000))


class ManualDriveApp:
    def __init__(self, track_seed: int = 42):
        track = make_track(track_seed, name=f"manual-{track_seed}")
        self.env = RacingEnv(track=track, max_steps=50000)
        self.root = tk.Tk()
        self.root.title("Minimum-Time Racing — Manual Environment")
        self.root.geometry(f"{WIDTH}x{HEIGHT}")
        self.root.configure(bg="#181818")
        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg="#181818", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.keys: set[str] = set()
        self.running = True
        self.root.bind_all("<KeyPress>", self._key_down)
        self.root.bind_all("<KeyRelease>", self._key_up)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.env.reset()
        self._draw_static_track()
        self.draw_dynamic()
        self.root.after(FRAME_MS, self.update)

    def _key_down(self, event):
        key = event.keysym.lower()
        fresh_press = key not in self.keys
        self.keys.add(key)
        if fresh_press and key == "r":
            self.env.reset()
            self._draw_static_track()
            self.draw_dynamic()
        elif key in ("q", "escape"):
            self.quit()

    def _key_up(self, event):
        self.keys.discard(event.keysym.lower())

    def _action(self) -> np.ndarray:
        # action[0] is normalized Delta-v; action[1] is normalized Delta-heading.
        if "w" in self.keys or "up" in self.keys:
            dv = 1.0
        elif "s" in self.keys or "down" in self.keys:
            dv = -1.0
        else:
            dv = 0.0
        if "a" in self.keys or "left" in self.keys:
            dtheta = 1.0
        elif "d" in self.keys or "right" in self.keys:
            dtheta = -1.0
        else:
            dtheta = 0.0
        return np.asarray([dv, dtheta], dtype=np.float32)

    @staticmethod
    def _screen(x, y):
        return WIDTH * 0.52 + x * SCALE, HEIGHT * 0.51 - y * SCALE

    def _coords(self, xs, ys):
        coords = []
        for x, y in zip(xs, ys):
            sx, sy = self._screen(float(x), float(y))
            coords.extend((sx, sy))
        return coords

    def _draw_static_track(self):
        self.canvas.delete("track")
        track = self.env.track
        center = self._coords(np.append(track.x, track.x[0]), np.append(track.y, track.y[0]))
        outer = self._coords(np.append(track.outer_x, track.outer_x[0]), np.append(track.outer_y, track.outer_y[0]))
        inner = self._coords(np.append(track.inner_x, track.inner_x[0]), np.append(track.inner_y, track.inner_y[0]))
        self.canvas.create_line(*center, fill="#383838", width=int(track.width * SCALE),
                                smooth=False, tags="track")
        self.canvas.create_line(*outer, fill="#eeeeee", width=3, tags="track")
        self.canvas.create_line(*inner, fill="#eeeeee", width=3, tags="track")
        self.canvas.create_line(*center, fill="#858585", width=1, tags="track")
        sx, sy = self._screen(track.x[0], track.y[0])
        self.canvas.create_oval(sx-5, sy-5, sx+5, sy+5, fill="#52b36c", outline="white", tags="track")
        self.canvas.tag_lower("track")

    def draw_dynamic(self):
        self.canvas.delete("dynamic")
        env = self.env
        track = env.track

        if len(env.trajectory_x) > 1:
            self.canvas.create_line(*self._coords(env.trajectory_x, env.trajectory_y),
                                    fill="#36a8eb", width=3, tags="dynamic")

        cx, cy = self._screen(env.x, env.y)
        for distance, angle_deg in zip(env.last_rays, RAY_ANGLES_DEG):
            angle = env.heading + np.deg2rad(angle_deg)
            ex, ey = self._screen(env.x + float(distance)*np.cos(angle),
                                  env.y + float(distance)*np.sin(angle))
            self.canvas.create_line(cx, cy, ex, ey, fill="#e7be48", width=1, tags="dynamic")

        radius = 8
        self.canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius,
                                fill="#e65353", outline="white", width=1, tags="dynamic")
        hx, hy = self._screen(env.x + 4.5*np.cos(env.heading), env.y + 4.5*np.sin(env.heading))
        self.canvas.create_line(cx, cy, hx, hy, fill="white", width=3, tags="dynamic")

        projection = env.last_info if env.last_info else {}
        fraction = float(np.clip(env.progress / track.length, 0.0, 1.0))
        if env.off_track:
            status = "OFF TRACK — PRESS R TO RESET"
        elif env.lap_complete:
            status = "LAP COMPLETE — PRESS R TO RESET"
        elif env.speed < 1e-4:
            status = "READY — HOLD W / UP TO ACCELERATE"
        else:
            status = "DRIVING"

        lines = [
            "RACING ENVIRONMENT — MANUAL CONTROL",
            "W / UP: increase speed | S / DOWN: decrease speed",
            "A / LEFT: change heading left | D / RIGHT: change heading right",
            "Release keys to hold current speed/heading change at zero; S brakes.",
            "R: reset    Q / ESC: quit",
            "",
            f"Status: {status}",
            f"Speed: {env.speed:6.2f} m/s    Sim time: {env.sim_time:6.2f} s",
            f"Lap progress: {100*fraction:5.1f}% of {track.length:.1f} m",
            f"Heading: {np.rad2deg(env.heading):6.1f} deg    Lateral offset: {projection.get('signed_lateral_offset', 0.0):6.2f} m",
            f"Actions: Delta-v max {MAX_DELTA_V:.3f} m/s/frame, Delta-heading max {np.rad2deg(MAX_DELTA_HEADING):.2f} deg/frame",
        ]
        self.canvas.create_rectangle(10, 10, 640, 250, fill="#222222", outline="#555555", tags="dynamic")
        y = 24
        for line in lines:
            self.canvas.create_text(22, y, anchor="w", text=line, fill="#eeeeee",
                                    font=("Consolas", 11), tags="dynamic")
            y += 20

    def update(self):
        if not self.running:
            return
        if not self.env.finished:
            _, _, terminated, truncated, _ = self.env.step(self._action())
            if terminated or truncated:
                if self.env.lap_complete:
                    print(f"Lap complete in {self.env.sim_time:.2f} simulated seconds. Press R to reset.")
                elif self.env.off_track:
                    print("Off track. Press R to reset.")
        self.draw_dynamic()
        if self.running:
            self.root.after(FRAME_MS, self.update)

    def quit(self):
        self.running = False
        self.env.close()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    ManualDriveApp().run()
