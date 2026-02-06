"""
GitHub Actions专用训练脚本
包含时间监控和模型上传功能
"""
import os
import sys
import time
import torch
import numpy as np
import random
from collections import deque
from chinese_chess import ChineseChess
from chinese_chess_ai import ChessTrainer, ChessMCTS


class TimeMonitor:
    """时间监控器"""
    def __init__(self, max_minutes=30):
        self.start_time = time.time()
        self.max_seconds = max_minutes * 60
    
    def elapsed_minutes(self):
        return (time.time() - self.start_time) / 60
    
    def is_time_exceeded(self):
        return (time.time() - self.start_time) > self.max_seconds
    
    def time_remaining_minutes(self):
        return max(0, (self.max_seconds - (time.time() - self.start_time)) / 60)


def train_with_monitor(red_model_path=None, black_model_path=None, 
                      epochs=10, games_per_epoch=3, steps_per_epoch=20,
                      max_time_minutes=30):
    """
    带时间监控的训练函数
    :param red_model_path: 红方模型路径（可选）
    :param black_model_path: 黑方模型路径（可选）
    :param epochs: 训练轮数
    :param games_per_epoch: 每轮对弈局数
    :param steps_per_epoch: 每轮训练步数
    :param max_time_minutes: 最大运行时间（分钟）
    """
    print(f"开始训练中国象棋AI...")
    print(f"最大允许时间: {max_time_minutes} 分钟")
    
    # 初始化时间监控器
    time_monitor = TimeMonitor(max_time_minutes)
    
    # 创建两个训练器，分别对应红方和黑方
    red_trainer = ChessTrainer()
    black_trainer = ChessTrainer()

    # 加载红方模型（如果提供了路径）
    if red_model_path and os.path.exists(red_model_path):
        try:
            red_trainer.load_model(red_model_path)
            print(f"红方已加载模型: {red_model_path}")
        except Exception as e:
            print(f"红方模型加载失败: {e}，使用随机模型")
    else:
        print("红方使用随机模型")

    # 加载黑方模型（如果提供了路径）
    if black_model_path and os.path.exists(black_model_path):
        try:
            black_trainer.load_model(black_model_path)
            print(f"黑方已加载模型: {black_model_path}")
        except Exception as e:
            print(f"黑方模型加载失败: {e}，使用随机模型")
    else:
        print("黑方使用随机模型")

    # 训练循环
    for epoch in range(epochs):
        print(f"\n=== 第 {epoch + 1}/{epochs} 轮训练 ===")
        
        # 检查时间是否超限
        if time_monitor.is_time_exceeded():
            print(f"时间超限 ({time_monitor.elapsed_minutes():.2f} 分钟)，停止训练")
            break

        # 自我对弈生成数据 - 使用两个不同的AI
        print("正在进行自我对弈 (红方 vs 黑方)...")

        for game_num in range(games_per_epoch):
            print(f"正在进行第 {game_num + 1}/{games_per_epoch} 局自我对弈...")

            # 检查时间是否超限
            if time_monitor.is_time_exceeded():
                print(f"时间超限 ({time_monitor.elapsed_minutes():.2f} 分钟)，停止当前轮次")
                break

            # 交替让不同AI执红先行，增加多样性
            if game_num % 2 == 0:
                red_ai = red_trainer
                black_ai = black_trainer
            else:
                red_ai = black_trainer  # 交换AI，让另一个执红
                black_ai = red_trainer

            game = ChineseChess()
            red_mcts = ChessMCTS(red_ai.net, num_simulations=75)
            black_mcts = ChessMCTS(black_ai.net, num_simulations=75)

            # 记录游戏数据
            game_states = []
            pis = []
            move_history_detailed = []
            current_player_ai = None
            current_mcts = None

            step = 0
            while not game.game_over:
                # 检查时间是否超限
                if time_monitor.is_time_exceeded():
                    print(f"时间超限 ({time_monitor.elapsed_minutes():.2f} 分钟)，停止当前游戏")
                    break

                # 显示详细信息
                red_pieces = np.count_nonzero(game.board > 0)
                black_pieces = np.count_nonzero(game.board < 0)
                print(f"第 {step + 1} 步 | 红方剩余棋子: {red_pieces} | 黑方剩余棋子: {black_pieces} | 当前玩家: {'红方' if game.current_player == 1 else '黑方'}")

                # 根据当前玩家选择对应的AI和MCTS
                if game.current_player == 1:  # 红方
                    current_player_ai = red_ai
                    current_mcts = red_mcts
                else:  # 黑方
                    current_player_ai = black_ai
                    current_mcts = black_mcts

                # 获取MCTS建议的移动
                moves, probs = current_mcts.get_action_prob(game, temp=1.0)  # 使用温度参数增加随机性

                if not moves or len(moves) == 0:
                    print("没有有效移动，游戏结束")
                    break

                # 根据概率选择移动（引入更多随机性）
                # 但优先选择获胜移动
                if len(moves) == 1 and len(probs) == 1:
                    # 这意味着这是唯一的获胜移动，直接选择
                    move_idx = 0
                else:
                    # 使用更多随机性，避免过于保守
                    dir_epsilon = 0.3  # 探索参数
                    dir_alpha = 0.2  # Dirichlet分布参数

                    if np.random.rand() < dir_epsilon:  # 有一定概率随机选择
                        # 添加Dirichlet噪声以增加探索
                        noise = np.random.dirichlet([dir_alpha] * len(probs))
                        # 将probs转换为numpy数组进行数学运算
                        probs_array = np.array(probs)
                        mixed_probs = (1 - dir_epsilon) * probs_array + dir_epsilon * noise
                        move_idx = np.random.choice(len(moves), p=mixed_probs)
                    elif np.random.rand() < 0.7:  # 70% 概率选择最高概率的移动
                        move_idx = np.argmax(probs)
                    else:  # 30% 概率根据概率分布随机选择
                        move_idx = np.random.choice(len(moves), p=probs)

                move = moves[move_idx]

                # 记录状态和策略
                state = game.get_state()
                pi = np.zeros(10 * 9)  # 位置概率分布
                from_pos, to_pos = move
                from_idx = from_pos[0] * 9 + from_pos[1]

                if from_idx < len(pi):
                    pi[from_idx] = probs[move_idx]

                game_states.append(state)
                pis.append(pi)

                # 记录移动详情：(移动, 移动前棋子, 被吃棋子)
                moving_piece = game.board[from_pos[0], from_pos[1]]
                captured_piece = game.board[to_pos[0], to_pos[1]]
                move_history_detailed.append((move, moving_piece, captured_piece))

                # 执行移动
                from_pos, to_pos = move
                success = game.make_move(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
                if not success:
                    print(f"移动失败: {from_pos} -> {to_pos}")
                    break

                step += 1

            # 游戏结束后，显示最终棋盘
            print("\n最终棋盘状态:")
            game.print_board()
            winner = game.winner if game.game_over else 0
            print(f"游戏结束! 总步数: {step}, 获胜方: {'红方' if winner == 1 else '黑方' if winner == -1 else '平局'}")

            # 为每个状态分配价值标签
            if winner is not None:
                values = []
                for i, state in enumerate(game_states):
                    current_player = 1 if i % 2 == 0 else -1  # 当前玩家

                    # 基础奖励
                    if winner == 0:  # 平局 - 更严厉惩罚，鼓励寻求胜负
                        base_value = -0.9  # 从-0.8改为-0.9，增加平局惩罚
                    elif winner == current_player:  # 获胜
                        base_value = 1.0
                    else:  # 失败
                        base_value = -1.0

                    # 吃子奖励
                    captured_value = 0
                    if i < len(move_history_detailed):
                        move_info = move_history_detailed[i]
                        captured_piece = move_info[2]  # 被吃的棋子
                        if captured_piece != 0 and (current_player > 0) == (captured_piece < 0):  # 当前玩家吃掉了对方棋子
                            captured_piece_type = abs(captured_piece)
                            # 根据棋子类型给予不同奖励
                            piece_rewards = {
                                1: 0.8,  # 将/帅
                                5: 0.4,  # 车
                                6: 0.3,  # 炮
                                4: 0.25, # 马
                                3: 0.2,  # 相/象
                                2: 0.15, # 士/仕
                                7: 0.1   # 兵/卒
                            }
                            captured_value = piece_rewards.get(captured_piece_type, 0.05)

                    # 检查是否将军（将对方将/帅置于被攻击状态）
                    check_value = 0
                    if i < len(game_states):  # 检查当前状态是否将军
                        # 从状态张量恢复棋盘
                        state_tensor = game_states[i]
                        temp_board = np.zeros((10, 9), dtype=np.int8)

                        # 从状态张量重建棋盘
                        for r in range(10):
                            for c in range(9):
                                for piece_type in range(7):  # 7种棋子类型
                                    if state_tensor[r, c, piece_type] == 1.0:  # 红方棋子
                                        temp_board[r, c] = piece_type + 1
                                    elif state_tensor[r, c, piece_type + 7] == 1.0:  # 黑方棋子
                                        temp_board[r, c] = -(piece_type + 1)

                        # 创建临时游戏实例来检查将军
                        temp_game = ChineseChess()
                        temp_game.board = temp_board
                        temp_game.current_player = current_player  # 当前玩家

                        # 检查当前玩家是否将军
                        opponent_king_pos = None
                        opponent_id = -current_player  # 对方ID
                        for r in range(10):
                            for c in range(9):
                                if temp_game.board[r, c] == opponent_id:  # 找到对方将/帅
                                    opponent_king_pos = (r, c)
                                    break
                            if opponent_king_pos:
                                break

                        if opponent_king_pos:
                            # 检查当前玩家是否有棋子能攻击到对方将/帅
                            for r in range(10):
                                for c in range(9):
                                    piece = temp_game.board[r, c]
                                    if piece != 0 and (piece > 0) == (current_player > 0):  # 当前玩家的棋子
                                        if temp_game.is_valid_move(r, c, opponent_king_pos[0], opponent_king_pos[1]):
                                            check_value = 0.1  # 将军奖励
                                            break
                                if check_value > 0:
                                    break

                    # 被吃惩罚（轻微）
                    eaten_value = 0
                    # 检查当前玩家的棋子是否在当前步被吃
                    current_move_info = move_history_detailed[i]
                    current_captured_piece = current_move_info[2]  # 当前步被吃的棋子
                    if current_captured_piece != 0 and (current_player > 0) == (current_captured_piece > 0):  # 当前玩家的棋子被吃
                        eaten_piece_type = abs(current_captured_piece)
                        piece_penalties = {
                            1: -0.5,  # 将/帅
                            5: -0.2,  # 车
                            6: -0.15, # 炮
                            4: -0.1,  # 马
                            3: -0.05, # 相/象
                            2: -0.03, # 士/仕
                            7: -0.01  # 兵/卒
                        }
                        eaten_value = piece_penalties.get(eaten_piece_type, -0.01)

                    # 综合价值
                    total_value = base_value + captured_value + eaten_value + check_value
                    values.append(total_value)

                # 将数据添加到对应AI的经验回放缓冲区中
                # 红方AI只学习红方视角的数据，黑方AI只学习黑方视角的数据
                for i, (state, pi, value) in enumerate(zip(game_states, pis, values)):
                    # 确定当前状态是由哪一方产生的
                    state_player = 1 if i % 2 == 0 else -1  # 假设第一个状态是红方的
                    if state_player == 1:  # 红方的状态
                        red_trainer.replay_buffer.append((state, pi, value))
                    else:  # 黑方的状态
                        black_trainer.replay_buffer.append((state, pi, value))

        # 检查时间是否超限
        if time_monitor.is_time_exceeded():
            print(f"时间超限 ({time_monitor.elapsed_minutes():.2f} 分钟)，停止训练")
            break

        # 训练网络 - 分别训练两个AI
        print("正在训练红方AI...")
        total_policy_loss_red = 0
        total_value_loss_red = 0
        total_loss_red = 0

        for step in range(steps_per_epoch):
            # 检查时间是否超限
            if time_monitor.is_time_exceeded():
                print(f"时间超限 ({time_monitor.elapsed_minutes():.2f} 分钟)，停止训练")
                break
                
            policy_loss, value_loss, loss = red_trainer.train_step()
            total_policy_loss_red += policy_loss
            total_value_loss_red += value_loss
            total_loss_red += loss

        if not time_monitor.is_time_exceeded():
            avg_policy_loss_red = total_policy_loss_red / steps_per_epoch if steps_per_epoch > 0 else 0
            avg_value_loss_red = total_value_loss_red / steps_per_epoch if steps_per_epoch > 0 else 0
            avg_total_loss_red = total_loss_red / steps_per_epoch if steps_per_epoch > 0 else 0

            print(f"红方平均损失: 策略={avg_policy_loss_red:.4f}, 价值={avg_value_loss_red:.4f}, 总计={avg_total_loss_red:.4f}")

        if not time_monitor.is_time_exceeded():
            print("正在训练黑方AI...")
            total_policy_loss_black = 0
            total_value_loss_black = 0
            total_loss_black = 0

            for step in range(steps_per_epoch):
                # 检查时间是否超限
                if time_monitor.is_time_exceeded():
                    print(f"时间超限 ({time_monitor.elapsed_minutes():.2f} 分钟)，停止训练")
                    break
                    
                policy_loss, value_loss, loss = black_trainer.train_step()
                total_policy_loss_black += policy_loss
                total_value_loss_black += value_loss
                total_loss_black += loss

            if not time_monitor.is_time_exceeded():
                avg_policy_loss_black = total_policy_loss_black / steps_per_epoch if steps_per_epoch > 0 else 0
                avg_value_loss_black = total_value_loss_black / steps_per_epoch if steps_per_epoch > 0 else 0
                avg_total_loss_black = total_loss_black / steps_per_epoch if steps_per_epoch > 0 else 0

                print(f"黑方平均损失: 策略={avg_policy_loss_black:.4f}, 价值={avg_value_loss_black:.4f}, 总计={avg_total_loss_black:.4f}")

        # 检查时间是否超限
        if time_monitor.is_time_exceeded():
            print(f"时间超限 ({time_monitor.elapsed_minutes():.2f} 分钟)，停止训练")
            break

        # 更新目标网络
        red_trainer.update_target_net()
        black_trainer.update_target_net()

        print(f"第 {epoch + 1} 轮训练完成")
        print(f"已用时间: {time_monitor.elapsed_minutes():.2f} 分钟, 剩余时间: {time_monitor.time_remaining_minutes():.2f} 分钟")

    # 保存最终模型 - 保存两个模型
    red_save_path = os.getenv("RED_MODEL_PATH", "red_model.pth")
    black_save_path = os.getenv("BLACK_MODEL_PATH", "black_model.pth")
    
    red_trainer.save_model(red_save_path)
    black_trainer.save_model(black_save_path)
    print(f"红方模型已保存至 {red_save_path}")
    print(f"黑方模型已保存至 {black_save_path}")
    
    print(f"训练完成! 总用时: {time_monitor.elapsed_minutes():.2f} 分钟")
    
    return red_save_path, black_save_path


def main():
    """主函数"""
    print("GitHub Actions中国象棋AI训练系统启动!")
    
    # 从环境变量获取参数
    red_model_path = os.getenv("RED_MODEL_INPUT_PATH")
    black_model_path = os.getenv("BLACK_MODEL_INPUT_PATH")
    epochs = int(os.getenv("EPOCHS", "5"))
    games_per_epoch = int(os.getenv("GAMES_PER_EPOCH", "3"))
    steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "20"))
    max_time_minutes = int(os.getenv("MAX_TIME_MINUTES", "30"))
    
    # 检查模型文件是否存在，如果不存在则设为None（使用随机模型）
    if red_model_path and not os.path.exists(red_model_path):
        print(f"红方模型文件不存在: {red_model_path}，将使用随机模型")
        red_model_path = None
        
    if black_model_path and not os.path.exists(black_model_path):
        print(f"黑方模型文件不存在: {black_model_path}，将使用随机模型")
        black_model_path = None
    
    print(f"训练参数:")
    print(f"- 红方模型路径: {red_model_path or '随机初始化'}")
    print(f"- 黑方模型路径: {black_model_path or '随机初始化'}")
    print(f"- 训练轮数: {epochs}")
    print(f"- 每轮对弈局数: {games_per_epoch}")
    print(f"- 每轮训练步数: {steps_per_epoch}")
    print(f"- 最大时间限制: {max_time_minutes} 分钟")
    
    # 开始训练
    red_path, black_path = train_with_monitor(
        red_model_path=red_model_path,
        black_model_path=black_model_path,
        epochs=epochs,
        games_per_epoch=games_per_epoch,
        steps_per_epoch=steps_per_epoch,
        max_time_minutes=max_time_minutes
    )
    
    print(f"训练完成! 模型已保存至: {red_path}, {black_path}")


if __name__ == "__main__":
    main()