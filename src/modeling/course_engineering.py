"""
Advanced course engineering system that quantifies course conditions into precise model inputs.
Transforms vague descriptions like "fast greens" into specific data points like "Stimpmeter 14.5".
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class CourseConditionData:
    """Quantified course condition data for modeling."""
    condition_name: str
    measurement_value: float
    measurement_unit: str
    difficulty_scale: float  # 1-10 scale
    player_skill_metric: str
    weight_in_model: float


@dataclass
class CourseSetup:
    """Complete engineered course setup for a specific tournament."""
    course_name: str
    tournament_name: str
    setup_date: str
    overall_difficulty: float  # Historical scoring average vs par
    conditions: List[CourseConditionData]
    weather_factors: Dict[str, float]
    setup_philosophy: str  # 'penal', 'strategic', 'heroic'


class CourseEngineeringSystem:
    """
    Converts course descriptions into precise, quantifiable model inputs.
    Implements the sophisticated approach of matching measurable course conditions
    to specific player skill metrics.
    """
    
    def __init__(self):
        self.course_database = {}
        self.player_skill_mappings = {}
        self.condition_weights = {}
        
        # Initialize US Open 2025 Oakmont setup
        self._initialize_oakmont_2025()
        
        # Initialize player skill mappings
        self._initialize_skill_mappings()
    
    def _initialize_oakmont_2025(self):
        """Initialize the precisely engineered Oakmont 2025 setup."""
        
        oakmont_conditions = [
            CourseConditionData(
                condition_name="green_speed",
                measurement_value=14.5,  # Stimpmeter reading
                measurement_unit="stimpmeter",
                difficulty_scale=9.5,
                player_skill_metric="sg_putt_fast_greens",
                weight_in_model=0.25
            ),
            CourseConditionData(
                condition_name="bunker_penalty",
                measurement_value=8.5,  # Composite bunker difficulty (1-10)
                measurement_unit="penalty_index",
                difficulty_scale=9.0,
                player_skill_metric="sand_save_percentage",
                weight_in_model=0.20
            ),
            CourseConditionData(
                condition_name="rough_height",
                measurement_value=4.5,  # Inches
                measurement_unit="inches",
                difficulty_scale=8.5,
                player_skill_metric="driving_accuracy",
                weight_in_model=0.30
            ),
            CourseConditionData(
                condition_name="course_length",
                measurement_value=7255,  # Total yardage
                measurement_unit="yards",
                difficulty_scale=7.5,
                player_skill_metric="driving_distance",
                weight_in_model=0.15
            ),
            CourseConditionData(
                condition_name="green_firmness",
                measurement_value=8.0,  # Firmness index (1-10)
                measurement_unit="firmness_index",
                difficulty_scale=8.0,
                player_skill_metric="sg_approach_firm_greens",
                weight_in_model=0.20
            ),
            CourseConditionData(
                condition_name="fairway_width",
                measurement_value=28,  # Average width in yards
                measurement_unit="yards",
                difficulty_scale=8.5,
                player_skill_metric="driving_accuracy",
                weight_in_model=0.25
            ),
            CourseConditionData(
                condition_name="pin_accessibility",
                measurement_value=3.0,  # Accessibility index (1-5, lower = harder)
                measurement_unit="accessibility_index",
                difficulty_scale=9.0,
                player_skill_metric="sg_approach_precision",
                weight_in_model=0.15
            )
        ]
        
        self.course_database["oakmont_2025"] = CourseSetup(
            course_name="Oakmont Country Club",
            tournament_name="US Open 2025",
            setup_date="2025-06-12",
            overall_difficulty=2.8,  # Historical US Open scoring average (+2.8 over par)
            conditions=oakmont_conditions,
            weather_factors={
                "wind_exposure": 0.8,  # Course exposure to wind (0-1)
                "drainage_quality": 0.9,  # How well course handles rain (0-1)
                "temperature_sensitivity": 0.3  # How much temp affects play (0-1)
            },
            setup_philosophy="penal"  # USGA philosophy: punish mistakes severely
        )
    
    def _initialize_skill_mappings(self):
        """Map course conditions to specific player skill metrics."""
        
        self.player_skill_mappings = {
            "green_speed": {
                "primary_metric": "sg_putt",
                "adjustment_factor": "fast_green_multiplier",
                "calculation": "base_putting_skill * fast_green_adjustment",
                "elite_threshold": 0.5,  # SG Putting threshold for "elite"
                "penalty_threshold": -0.3  # Below this = significant penalty
            },
            "bunker_penalty": {
                "primary_metric": "sand_save_percentage", 
                "adjustment_factor": "bunker_difficulty_multiplier",
                "calculation": "sand_save_rate * bunker_penalty_factor",
                "elite_threshold": 60,  # 60%+ sand save rate
                "penalty_threshold": 40   # Below 40% = major penalty
            },
            "rough_height": {
                "primary_metric": "driving_accuracy",
                "adjustment_factor": "rough_penalty_multiplier", 
                "calculation": "accuracy_rate * rough_avoidance_bonus",
                "elite_threshold": 70,  # 70%+ fairways hit
                "penalty_threshold": 55   # Below 55% = major penalty
            },
            "course_length": {
                "primary_metric": "driving_distance",
                "adjustment_factor": "length_advantage_multiplier",
                "calculation": "distance_advantage * length_reward_factor",
                "elite_threshold": 300,  # 300+ yard average
                "penalty_threshold": 270   # Below 270 = disadvantage
            },
            "green_firmness": {
                "primary_metric": "sg_app",
                "adjustment_factor": "firm_green_multiplier",
                "calculation": "approach_skill * firm_condition_factor",
                "elite_threshold": 0.3,  # SG Approach threshold
                "penalty_threshold": -0.2  # Below this = penalty
            }
        }
    
    def calculate_player_course_fit(self, player_data: Dict, 
                                  course_setup: CourseSetup) -> Dict:
        """
        Calculate precise player-course fit based on engineered conditions.
        This is the core function that matches player skills to course demands.
        """
        
        fit_scores = {}
        total_weighted_score = 0
        total_weights = 0
        
        for condition in course_setup.conditions:
            condition_name = condition.condition_name
            
            if condition_name in self.player_skill_mappings:
                mapping = self.player_skill_mappings[condition_name]
                
                # Get player's skill level for this condition
                player_skill = self._extract_player_skill(
                    player_data, mapping["primary_metric"]
                )
                
                # Calculate condition-specific fit score
                fit_score = self._calculate_condition_fit(
                    player_skill, condition, mapping
                )
                
                # Weight the score
                weighted_score = fit_score * condition.weight_in_model
                
                fit_scores[condition_name] = {
                    "raw_skill": player_skill,
                    "condition_difficulty": condition.difficulty_scale,
                    "fit_score": fit_score,
                    "weighted_score": weighted_score,
                    "weight": condition.weight_in_model
                }
                
                total_weighted_score += weighted_score
                total_weights += condition.weight_in_model
        
        # Calculate overall course fit
        overall_fit = total_weighted_score / total_weights if total_weights > 0 else 0
        
        return {
            "player_name": player_data.get("player_name", "Unknown"),
            "course_name": course_setup.course_name,
            "overall_fit_score": overall_fit,
            "condition_breakdown": fit_scores,
            "fit_category": self._categorize_fit(overall_fit),
            "key_advantages": self._identify_advantages(fit_scores),
            "key_vulnerabilities": self._identify_vulnerabilities(fit_scores)
        }
    
    def _extract_player_skill(self, player_data: Dict, metric: str) -> float:
        """Extract specific skill metric from player data."""
        
        # Map our metric names to actual data columns
        metric_mappings = {
            "sg_putt": "sg_putt",
            "sand_save_percentage": "sand_save_pct",  # Would need this data
            "driving_accuracy": "driving_acc", 
            "driving_distance": "driving_dist",
            "sg_app": "sg_app"
        }
        
        actual_column = metric_mappings.get(metric, metric)
        
        if actual_column in player_data:
            return float(player_data[actual_column])
        else:
            # Provide reasonable defaults based on player ranking
            rank = player_data.get("datagolf_rank", 100)
            return self._estimate_skill_from_rank(metric, rank)
    
    def _estimate_skill_from_rank(self, metric: str, rank: int) -> float:
        """Estimate skill level based on player ranking when specific data unavailable."""
        
        # Elite players (top 20) get better estimates
        if rank <= 20:
            multiplier = 1.2
        elif rank <= 50:
            multiplier = 1.0
        elif rank <= 100:
            multiplier = 0.8
        else:
            multiplier = 0.6
        
        base_estimates = {
            "sg_putt": 0.2 * multiplier,
            "sand_save_percentage": 55 * multiplier,
            "driving_accuracy": 65 * multiplier,
            "driving_distance": 285 * multiplier,
            "sg_app": 0.1 * multiplier
        }
        
        return base_estimates.get(metric, 0)
    
    def _calculate_condition_fit(self, player_skill: float, 
                               condition: CourseConditionData,
                               mapping: Dict) -> float:
        """Calculate how well a player's skill matches a specific course condition."""
        
        elite_threshold = mapping["elite_threshold"]
        penalty_threshold = mapping["penalty_threshold"]
        
        # Normalize skill relative to thresholds
        if player_skill >= elite_threshold:
            # Elite performance - gets bonus
            skill_factor = 1.0 + (player_skill - elite_threshold) / elite_threshold * 0.5
        elif player_skill >= penalty_threshold:
            # Average performance - neutral
            skill_factor = 0.8 + (player_skill - penalty_threshold) / (elite_threshold - penalty_threshold) * 0.4
        else:
            # Below threshold - penalty
            skill_factor = 0.3 + (player_skill / penalty_threshold) * 0.5
        
        # Adjust based on condition difficulty
        difficulty_factor = 1.0 + (condition.difficulty_scale - 5) / 10
        
        # Final fit score (0-2 scale, 1.0 = perfect fit)
        fit_score = skill_factor / difficulty_factor
        
        return max(0.1, min(2.0, fit_score))  # Clamp between 0.1 and 2.0
    
    def _categorize_fit(self, overall_fit: float) -> str:
        """Categorize overall course fit."""
        if overall_fit >= 0.85:
            return "Excellent Fit"
        elif overall_fit >= 0.75:
            return "Good Fit"
        elif overall_fit >= 0.65:
            return "Average Fit"
        elif overall_fit >= 0.55:
            return "Poor Fit"
        else:
            return "Very Poor Fit"
    
    def _identify_advantages(self, fit_scores: Dict) -> List[str]:
        """Identify player's key advantages for this course."""
        advantages = []
        
        for condition, scores in fit_scores.items():
            if scores["fit_score"] >= 1.2:
                advantages.append(f"Strong {condition.replace('_', ' ')}")
        
        return advantages[:3]  # Top 3 advantages
    
    def _identify_vulnerabilities(self, fit_scores: Dict) -> List[str]:
        """Identify player's key vulnerabilities for this course."""
        vulnerabilities = []
        
        for condition, scores in fit_scores.items():
            if scores["fit_score"] <= 0.8:
                vulnerabilities.append(f"Weak {condition.replace('_', ' ')}")
        
        return vulnerabilities[:3]  # Top 3 vulnerabilities
    
    def generate_course_report(self, course_setup: CourseSetup) -> str:
        """Generate detailed course setup report."""
        
        report = f"""
COURSE ENGINEERING REPORT
{course_setup.course_name} - {course_setup.tournament_name}
Setup Date: {course_setup.setup_date}
Overall Difficulty: {course_setup.overall_difficulty:+.1f} over par

ENGINEERED CONDITIONS:
"""
        
        for condition in course_setup.conditions:
            report += f"""
{condition.condition_name.replace('_', ' ').title()}:
  Measurement: {condition.measurement_value} {condition.measurement_unit}
  Difficulty Scale: {condition.difficulty_scale}/10
  Model Weight: {condition.weight_in_model:.1%}
  Player Skill Required: {condition.player_skill_metric}
"""
        
        return report


if __name__ == "__main__":
    print("Course Engineering System initialized")
    print("Ready to convert course conditions into precise model inputs")
