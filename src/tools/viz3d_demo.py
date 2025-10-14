#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PRIME VERSION: Framework de visualização de alta fidelidade para a fusão
de um sistema binário de buracos negros.

Este script implementa um modelo de inspiração Pós-Newtoniano, um evento de
fusão com perda de massa, e uma fase de "ringdown" pós-fusão para máxima
precisão visual e física.

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    import pyvista as pv
except ImportError:
    logging.error("Dependências não encontradas. Por favor, instale com: pip install pyvista imageio imageio-ffmpeg tqdm")
    raise

# --- Configuração Centralizada ---
@dataclass
class VisualizationConfig:
    """Agrupa todos os parâmetros configuráveis para a visualização."""
    # Saída
    output_path: Path = Path("results/viz3d/")

    # Renderização
    resolution: Tuple[int, int] = (720, 440)
    fps: int = 90
    n_frames: int = 10000
    background_color: str = "#0a0a14"
    enable_camera_orbit: bool = False

    # Grade do Espaço-Tempo
    grid_size: float = 10.0
    grid_lines: int = 81
    line_width: float = 1.0
    line_color: str = "white"

    # FATOR DE INTENSIDADE GLOBAL (REDUZIDO)
    wave_aggression_factor: float = 1.0 # Reduzido de 4.5 para 1.0

    # Onda Gravitacional
    z_scale: float = 0.5 # Reduzido de 1.8 para 0.5
    polarization_angle_deg: float = 10.0
    decay_rate: float = 0.10
    ramp_up_duration: float = 0.10
    visible_waves_on_grid: float = 3.0

    # Sistema Binário Físico
    mass_bh1: float = 30.0
    mass_bh2: float = 25.0
    bh_r_initial_separation: float = 6.0
    bh_r_final_separation: float = 0.50
    bh_radius_factor: float = 0.04
    bh_color: str = "#000000"
    merger_time_factor: float = 0.95
    bh_base_angular_speed: float = 20.0

    # Câmera
    cam_pos: Tuple[float, float, float] = (16.0, -14.0, 16.0)
    cam_focal: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    cam_up: Tuple[float, float, float] = (0.0, 0.0, 1.0)


def ease_in_out_cubic(s: float) -> float:
    s = np.clip(s, 0.0, 1.0)
    return 3 * s**2 - 2 * s**3


class GravitationalWaveVisualizer:
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.config.z_scale *= self.config.wave_aggression_factor # z_scale final será 0.5 * 1.0 = 0.5
        self._setup_environment()
        
        self.plotter: pv.Plotter | None = None
        self.actors: Dict[str, Any] = {}
        self.grid_mesh: pv.PolyData | None = None
        self.grid_points_flat: np.ndarray | None = None
        self.R_flat: np.ndarray | None = None
        self.e_plus: np.ndarray | None = None
        self.e_cross: np.ndarray | None = None
        
        self.merger_happened = False
        self.merger_frame = -1
        self.current_angle = 0.0
        self.inspiral_constant = self._calculate_inspiral_constant()

    def _setup_environment(self) -> None:
        logging.info("Configurando ambiente de visualização.")
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        import os
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

    def _initialize_plotter(self) -> None:
        logging.info("Inicializando plotter PyVista.")
        pv.global_theme.background = self.config.background_color
        self.plotter = pv.Plotter(window_size=self.config.resolution, off_screen=True)
        self.plotter.camera_position = [self.config.cam_pos, self.config.cam_focal, self.config.cam_up]

    def _create_scene_actors(self) -> None:
        logging.info("Criando atores da cena.")
        s, n = self.config.grid_size, self.config.grid_lines
        coords = np.linspace(-s, s, n)
        self.grid_mesh = self._create_line_grid_mesh(coords)
        self.grid_points_flat = self.grid_mesh.points.copy()
        
        self.actors['grid'] = self.plotter.add_mesh(self.grid_mesh, color=self.config.line_color, line_width=self.config.line_width, style='wireframe')

        X, Y = self.grid_points_flat[:, 0], self.grid_points_flat[:, 1]
        self.R_flat = np.sqrt(X**2 + Y**2)
        
        m1, m2 = self.config.mass_bh1, self.config.mass_bh2
        total_mass = m1 + m2
        eta = (m1 * m2) / (total_mass**2) # Parâmetro de massa simétrica
        
        bh1_rad = self.config.bh_radius_factor * m1**0.33
        bh2_rad = self.config.bh_radius_factor * m2**0.33
        
        # Perda de massa na fusão (aproximação)
        mass_loss_factor = 0.05 # Perda de ~5% da massa como energia
        final_mass = total_mass * (1 - mass_loss_factor * (eta / 0.25))
        merged_rad = self.config.bh_radius_factor * final_mass**0.33

        r1_init = (m2 / total_mass) * self.config.bh_r_initial_separation
        r2_init = (m1 / total_mass) * self.config.bh_r_initial_separation

        self.actors['bh1'] = self.plotter.add_mesh(pv.Sphere(radius=bh1_rad, center=(r1_init, 0, 0.1)), color=self.config.bh_color, specular=1.0, specular_power=15)
        self.actors['bh2'] = self.plotter.add_mesh(pv.Sphere(radius=bh2_rad, center=(-r2_init, 0, 0.1)), color=self.config.bh_color, specular=1.0, specular_power=15)
        self.actors['merged_bh'] = self.plotter.add_mesh(pv.Sphere(radius=merged_rad, center=(0, 0, 0.1)), color=self.config.bh_color, specular=1.0, specular_power=15)
        self.actors['merged_bh'].visibility = False
        
        # Efeito de brilho (glow) para a fusão
        self.actors['merger_glow'] = self.plotter.add_mesh(pv.Sphere(radius=merged_rad * 1.5), color="yellow", opacity=0.0)
        
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

    def _calculate_inspiral_constant(self) -> float:
        r0_4 = self.config.bh_r_initial_separation**4
        rf_4 = self.config.bh_r_final_separation**4
        return (r0_4 - rf_4) / self.config.merger_time_factor

    def _calculate_wave_at_time(self, t01: float, current_orbital_freq: float) -> Tuple[np.ndarray, float]:
        freq_gw = current_orbital_freq * 2.0
        amp = 0.0

        if not self.merger_happened: # Fase de Inspiral
            normalized_t_merger = t01 / self.config.merger_time_factor
            amp_raw = np.interp(normalized_t_merger, [0, 0.8, 0.95, 1.0], [0.1, 0.5, 1.0, 1.5])
            amp = amp_raw
            if t01 < self.config.ramp_up_duration:
                amp *= ease_in_out_cubic(t01 / self.config.ramp_up_duration)
        else: # Fase de Ringdown
            t_after_merger = (t01 - self.config.merger_time_factor)
            
            # Frequência de ringdown (aproximada)
            final_mass = self.config.mass_bh1 + self.config.mass_bh2
            ringdown_freq = 1500 / final_mass # Simplificação
            freq_gw = ringdown_freq

            # Amplitude do ringdown (decaimento exponencial)
            ringdown_decay_time = 0.05
            amp_ringdown = 1.5 * np.exp(-t_after_merger / ringdown_decay_time)
            amp = amp_ringdown

        # Aplicar o wave_aggression_factor ao final da amplitude calculada
        amp *= self.config.wave_aggression_factor

        visual_wavelength = self.config.grid_size / self.config.visible_waves_on_grid
        propagation_speed = visual_wavelength * freq_gw
        phase = 2.0 * np.pi * freq_gw * (t01 - self.R_flat / (propagation_speed + 1e-9))
        decay = 1.0 / (1.0 + self.config.decay_rate * self.R_flat)
        h = (amp * decay) * (np.cos(phase) * self.e_plus + np.sin(phase) * self.e_cross)
        return h, freq_gw

    def _get_binary_state(self, t01: float, dt_frame: float) -> Tuple[float, float, float]:
        if self.merger_happened:
            return 0.0, self.current_angle, 0.0
        
        radius_4 = self.config.bh_r_initial_separation**4 - self.inspiral_constant * t01
        total_separation = radius_4**0.25
        
        angular_speed_factor = (self.config.bh_r_initial_separation / (total_separation + 1e-9))**1.5
        current_angular_speed = self.config.bh_base_angular_speed * angular_speed_factor
        
        self.current_angle += current_angular_speed * dt_frame
        
        orbital_frequency = current_angular_speed / (2 * np.pi)
        return total_separation, self.current_angle, orbital_frequency

    def _update_frame(self, frame_idx: int, t01: float) -> None:
        dt_frame = 1.0 / max(1, self.config.n_frames - 1)
        
        total_separation, current_angle, current_orbital_freq = self._get_binary_state(t01, dt_frame)

        h_values, current_gw_freq = self._calculate_wave_at_time(t01, current_orbital_freq)
        updated_points = self.grid_points_flat.copy()
        updated_points[:, 2] = self.config.z_scale * h_values
        self.grid_mesh.points = updated_points
        
        m1, m2, total_mass = self.config.mass_bh1, self.config.mass_bh2, self.config.mass_bh1 + self.config.mass_bh2

        if not self.merger_happened and total_separation <= self.config.bh_r_final_separation:
            logging.info(f"Fusão detectada no frame {frame_idx} (t={t01:.2f})!")
            self.merger_happened = True
            self.merger_frame = frame_idx
            self.actors['bh1'].visibility = False
            self.actors['bh2'].visibility = False
            self.actors['merged_bh'].visibility = True

        if not self.merger_happened:
            r1 = (m2 / total_mass) * total_separation
            r2 = (m1 / total_mass) * total_separation
            pos1 = (r1 * np.cos(current_angle), r1 * np.sin(current_angle), 0.1)
            pos2 = (-r2 * np.cos(current_angle), -r2 * np.sin(current_angle), 0.1)
            self.actors['bh1'].SetPosition(pos1)
            self.actors['bh2'].SetPosition(pos2)
        
        if self.merger_happened:
            t_since_merger = (frame_idx - self.merger_frame) / self.config.fps
            glow_duration = 0.5
            if t_since_merger < glow_duration:
                glow_progress = t_since_merger / glow_duration
                glow_opacity = ease_in_out_cubic(1 - glow_progress)
                glow_scale = 1 + 2 * glow_progress
                self.actors['merger_glow'].GetProperty().SetOpacity(glow_opacity)
                self.actors['merger_glow'].SetScale(glow_scale, glow_scale, glow_scale)
            else:
                self.actors['merger_glow'].GetProperty().SetOpacity(0.0)

        self.plotter.remove_actor(self.actors['time_text'])
        status = "Ringdown" if self.merger_happened else "Inspiral"
        new_text = f"Status: {status} | t={t01:.2f} | F_gw={current_gw_freq:.1f} Hz"
        self.actors['time_text'] = self.plotter.add_text(new_text, font_size=10, position="upper_left", color=self.config.line_color)
        
        if self.config.enable_camera_orbit:
            self.plotter.camera.azimuth += 0.1

    def run_animation(self) -> None:
        self._initialize_plotter()
        self._create_scene_actors()
        self._precompute_polarization_basis()
        
        output_filename = self.config.output_path / f"merger_{self.config.mass_bh1}M_{self.config.mass_bh2}M.mp4"
        logging.info(f"Iniciando renderização para: {output_filename}")
        self.plotter.open_movie(str(output_filename), framerate=self.config.fps)
        self.plotter.show(auto_close=False, interactive=False)

        for k in tqdm(range(self.config.n_frames), desc="Renderizando frames"):
            t01 = k / max(self.config.n_frames - 1, 1)
            self._update_frame(k, t01)
            self.plotter.write_frame()

        self.plotter.close()
        logging.info(f"Animação concluída! Salva em: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Gerador de Simulação de Fusão de Buracos Negros.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    resolutions = {"720p": (1280, 720), "1080p": (1920, 1080), "4K": (3840, 2160)}
    
    parser.add_argument("--output", type=Path, default=VisualizationConfig.output_path)
    parser.add_argument("--resolution", type=str, default="1080p", choices=resolutions.keys())
    parser.add_argument("--frames", type=int, default=VisualizationConfig.n_frames)
    parser.add_argument("--m1", type=float, default=VisualizationConfig.mass_bh1, help="Massa do primeiro BH (massas solares).")
    parser.add_argument("--m2", type=float, default=VisualizationConfig.mass_bh2, help="Massa do segundo BH (massas solares).")
    parser.add_argument("--angular-speed", type=float, default=VisualizationConfig.bh_base_angular_speed, help="Velocidade angular base da órbita.")
    parser.add_argument("--waves", type=float, default=VisualizationConfig.visible_waves_on_grid, help="Número de ondas visíveis do centro à borda.")
    parser.add_argument("--orbit-camera", action="store_true", help="Ativa a órbita da câmera.")
    
    args = parser.parse_args()

    config = VisualizationConfig(
        output_path=args.output,
        resolution=resolutions[args.resolution],
        n_frames=args.frames,
        mass_bh1=args.m1,
        mass_bh2=args.m2,
        bh_base_angular_speed=args.angular_speed,
        visible_waves_on_grid=args.waves,
        enable_camera_orbit=args.orbit_camera,
    )
    
    visualizer = GravitationalWaveVisualizer(config)
    visualizer.run_animation()

if __name__ == "__main__":
    main()