#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Framework de visualização para demonstrar a deformação AGRESSIVA de uma
grade 2D do espaço-tempo por uma onda gravitacional contínua.

Este script foca na CLAREZA da PROPAGAÇÃO da onda, com um início suave
para mostrar a onda nascendo no centro e se expandindo para fora.
A câmera é estática por padrão para uma observação científica clara.

Controlado via argumentos de linha de comando. Use 'python viz_gw.py --help'.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
from tqdm import tqdm

# Configuração do logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    import pyvista as pv
except ImportError:
    logging.error(
        "Dependências não encontradas. Por favor, instale com: "
        "pip install pyvista imageio imageio-ffmpeg tqdm"
    )
    raise

# --- Configuração Centralizada ---
@dataclass
class VisualizationConfig:
    """Agrupa todos os parâmetros configuráveis para a visualização."""
    # Saída
    output_path: Path = Path("results/viz3d/wave_propagation_final.mp4")

    # Renderização
    resolution: Tuple[int, int] = (720, 440)
    fps: int = 60
    n_frames: int = 1000
    background_color: str = "#050510"
    enable_camera_orbit: bool = False

    # Grade do Espaço-Tempo
    grid_size: float = 100.0
    grid_lines: int = 1000
    line_width: float = 1.0
    line_color: str = "white"

    # FATOR DE INTENSIDADE GLOBAL
    wave_aggression_factor: float = 1.5

    # Onda Gravitacional
    z_scale: float = 2.0
    f_initial: float = 2.0
    f_final: float = 40.0
    a_initial: float = 0.0
    a_final: float = 1.0
    polarization_angle_deg: float = 30.0
    decay_rate: float = 0.10
    ramp_up_duration: float = 0.20

    # PARÂMETRO-CHAVE para controle visual
    visible_waves_on_grid: float = 2.5

    # Sistema Binário
    bh_r_initial: float = 3.0
    bh_r_final: float = 0.5
    bh_radius: float = 0.20
    bh_z_offset: float = 0.15
    bh_color: str = "#050505"
    bh_angular_exponent: float = 1.7

    # Câmera
    cam_pos: Tuple[float, float, float] = (15.0, -13.0, 15.0)
    cam_focal: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    cam_up: Tuple[float, float, float] = (0.0, 0.0, 1.0)


def ease_in_out_cubic(s: float) -> float:
    s = np.clip(s, 0.0, 1.0)
    return 3 * s**2 - 2 * s**3


class GravitationalWaveVisualizer:
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.config.z_scale *= self.config.wave_aggression_factor
        self._setup_environment()
        
        self.plotter: pv.Plotter | None = None
        self.actors: Dict[str, Any] = {}
        self.grid_mesh: pv.PolyData | None = None
        self.grid_points_flat: np.ndarray | None = None
        self.R_flat: np.ndarray | None = None
        self.e_plus: np.ndarray | None = None
        self.e_cross: np.ndarray | None = None
        
    def _setup_environment(self) -> None:
        logging.info("Configurando ambiente de visualização.")
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        import os
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

    def _initialize_plotter(self) -> None:
        logging.info("Inicializando plotter PyVista.")
        pv.global_theme.background = self.config.background_color
        self.plotter = pv.Plotter(window_size=self.config.resolution, off_screen=True)
        self.plotter.camera_position = [
            self.config.cam_pos, self.config.cam_focal, self.config.cam_up
        ]

    def _create_scene_actors(self) -> None:
        logging.info("Criando atores da cena.")
        s = self.config.grid_size
        n = self.config.grid_lines
        coords = np.linspace(-s, s, n)
        self.grid_mesh = self._create_line_grid_mesh(coords)
        self.grid_points_flat = self.grid_mesh.points.copy()
        
        self.actors['grid'] = self.plotter.add_mesh(
            self.grid_mesh, 
            color=self.config.line_color, 
            line_width=self.config.line_width,
            style='wireframe'
        )

        X, Y = self.grid_points_flat[:, 0], self.grid_points_flat[:, 1]
        self.R_flat = np.sqrt(X**2 + Y**2)
        
        (bx1, by1), _ = self._get_binary_positions(0.0)
        z_pos = self.config.bh_z_offset
        self.actors['bh1'] = self.plotter.add_mesh(pv.Sphere(radius=self.config.bh_radius, center=(bx1, by1, z_pos)), color=self.config.bh_color, specular=1.0, specular_power=10)
        self.actors['bh2'] = self.plotter.add_mesh(pv.Sphere(radius=self.config.bh_radius, center=(-bx1, -by1, z_pos)), color=self.config.bh_color, specular=1.0, specular_power=10)

        self.plotter.add_mesh(pv.Sphere(radius=self.config.bh_r_final * 0.8), color=self.config.line_color, opacity=0.1, style='wireframe')
        self.actors['time_text'] = self.plotter.add_text("", font_size=10, position="upper_left", color=self.config.line_color)
    
    def _create_line_grid_mesh(self, coords: np.ndarray) -> pv.PolyData:
        grid = pv.MultiBlock()
        s = coords[-1]
        for y in coords: grid.append(pv.Line([-s, y, 0], [s, y, 0]))
        for x in coords: grid.append(pv.Line([x, -s, 0], [x, s, 0]))
        return grid.combine()

    def _precompute_polarization_basis(self) -> None:
        angle_rad = np.deg2rad(self.config.polarization_angle_deg)
        ct, st = np.cos(angle_rad), np.sin(angle_rad)
        X, Y = self.grid_points_flat[:, 0], self.grid_points_flat[:, 1]
        xp, yp = ct * X + st * Y, -st * X + ct * Y
        r_squared = xp**2 + yp**2 + 1e-12
        self.e_plus = (xp**2 - yp**2) / r_squared
        self.e_cross = (2.0 * xp * yp) / r_squared

    def _calculate_wave_at_time(self, t01: float) -> Tuple[np.ndarray, float]:
        s = ease_in_out_cubic(t01)
        freq = self.config.f_initial + (self.config.f_final - self.config.f_initial) * s
        amp = self.config.a_initial + (self.config.a_final - self.config.a_initial) * s
        amp *= self.config.wave_aggression_factor

        if t01 < self.config.ramp_up_duration:
            ramp_up_factor = ease_in_out_cubic(t01 / self.config.ramp_up_duration)
            amp *= ramp_up_factor

        visual_wavelength = self.config.grid_size / self.config.visible_waves_on_grid
        propagation_speed = visual_wavelength * freq

        phase = 2.0 * np.pi * freq * (t01 - self.R_flat / (propagation_speed + 1e-9))
        decay = 1.0 / (1.0 + self.config.decay_rate * self.R_flat)
        h_plus = amp * decay * np.cos(phase)
        h_cross = amp * decay * np.sin(phase)
        h = h_plus * self.e_plus + h_cross * self.e_cross
        return h, freq

    def _get_binary_positions(self, t01: float) -> Tuple[Tuple, Tuple]:
        s = ease_in_out_cubic(t01)
        radius = self.config.bh_r_initial + (self.config.bh_r_final - self.config.bh_r_initial) * s
        angle = 8.0 * np.pi * (t01 ** self.config.bh_angular_exponent)
        pos1 = (radius * np.cos(angle), radius * np.sin(angle))
        pos2 = (-pos1[0], -pos1[1])
        return pos1, pos2

    def _update_frame(self, t01: float) -> None:
        h_values, current_freq = self._calculate_wave_at_time(t01)
        updated_points = self.grid_points_flat.copy()
        updated_points[:, 2] = self.config.z_scale * h_values
        self.grid_mesh.points = updated_points
        
        (bx1, by1), (bx2, by2) = self._get_binary_positions(t01)
        z_pos = self.config.bh_z_offset
        self.actors['bh1'].SetPosition((bx1, by1, z_pos))
        self.actors['bh2'].SetPosition((bx2, by2, z_pos))
        
        self.plotter.remove_actor(self.actors['time_text'])
        new_text = f"Tempo: {t01:.2f} | Frequência: {current_freq:.1f} Hz"
        self.actors['time_text'] = self.plotter.add_text(new_text, font_size=10, position="upper_left", color=self.config.line_color)
        
        if self.config.enable_camera_orbit:
            self.plotter.camera.azimuth += 0.1

    def run_animation(self) -> None:
        self._initialize_plotter()
        self._create_scene_actors()
        self._precompute_polarization_basis()

        logging.info(f"Iniciando renderização de {self.config.n_frames} frames...")
        self.plotter.open_movie(str(self.config.output_path), framerate=self.config.fps)
        self.plotter.show(auto_close=False, interactive=False)

        for k in tqdm(range(self.config.n_frames), desc="Renderizando frames"):
            t01 = k / max(self.config.n_frames - 1, 1)
            self._update_frame(t01)
            self.plotter.write_frame()

        self.plotter.close()
        logging.info(f"Animação concluída! Salva em: {self.config.output_path}")

def main():
    parser = argparse.ArgumentParser(description="Gerador de Animação de Deformação do Espaço-Tempo.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # --- CORREÇÃO APLICADA AQUI ---
    # Corrigido o erro de digitação de "100p" para "1080p"
    resolutions = {"720p": (1280, 720), "1080p": (1920, 1080), "4K": (3840, 2160)}
    
    parser.add_argument("--output", type=Path, default=VisualizationConfig.output_path)
    parser.add_argument("--resolution", type=str, default="1080p", choices=resolutions.keys())
    parser.add_argument("--frames", type=int, default=VisualizationConfig.n_frames)
    parser.add_argument("--aggression", type=float, default=VisualizationConfig.wave_aggression_factor)
    parser.add_argument("--waves", type=float, default=VisualizationConfig.visible_waves_on_grid, help="Número de ondas visíveis do centro à borda.")
    parser.add_argument("--orbit-camera", action="store_true", help="Ativa a órbita da câmera (desabilitado por padrão).")
    
    args = parser.parse_args()

    config = VisualizationConfig(
        output_path=args.output,
        resolution=resolutions[args.resolution],
        n_frames=args.frames,
        wave_aggression_factor=args.aggression,
        visible_waves_on_grid=args.waves,
        enable_camera_orbit=args.orbit_camera,
    )
    
    visualizer = GravitationalWaveVisualizer(config)
    visualizer.run_animation()

if __name__ == "__main__":
    main()