"""
中国象棋AI主程序
使用强化学习训练中国象棋AI
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from chinese_chess import ChineseChess


class ChessNetwork(nn.Module):
    """中国象棋神经网络，同时输出策略和价值"""
    
    def __init__(self, board_height=10, board_width=9, num_pieces=15):  # 更新为15个平面
        super(ChessNetwork, self).__init__()
        
        # 输入层：15个平面代表不同的棋子类型 + 当前玩家
        self.conv_input = nn.Conv2d(num_pieces, 64, kernel_size=3, padding=1)
        
        # 残差块
        self.res_blocks = nn.ModuleList([self._residual_block(64) for _ in range(6)])
        
        # 策略头
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * board_height * board_width, board_height * board_width)  # 输出棋盘上每个位置的选择概率
        
        # 价值头
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_height * board_width, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
    def _residual_block(self, channels):
        """残差块"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        # x shape: (batch, 15, 10, 9) - 15 planes: 7 red pieces + 7 black pieces + 1 current player
        batch_size = x.size(0)

        # 输入处理
        out = nn.functional.relu(self.conv_input(x))

        # 残差层
        for res_block in self.res_blocks:
            residual = out
            out = res_block(out)
            out += residual
            out = nn.functional.relu(out)

        # 策略头
        policy = nn.functional.relu(self.policy_conv(out))
        policy = policy.reshape(batch_size, -1)  # 使用reshape而不是view
        policy = self.policy_fc(policy)
        policy = nn.functional.softmax(policy, dim=1)

        # 价值头
        value = nn.functional.relu(self.value_conv(out))
        value = value.reshape(batch_size, -1)  # 使用reshape而不是view
        value = nn.functional.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)).squeeze(1)

        return policy, value


class ChessMCTSNode:
    """中国象棋MCTS节点"""
    
    def __init__(self, parent=None, prior_prob=1.0, move=None):
        self.parent = parent
        self.children = {}  # {(from_pos, to_pos): ChessMCTSNode}
        self.move = move  # ((from_r, from_c), (to_r, to_c))
        self.prior_prob = prior_prob
        self.visit_count = 0
        self.total_value = 0.0
        self.q_value = 0.0
        self.u_value = 0.0
    
    def ucb_score(self, exploration_param=1.0):
        if self.visit_count == 0:
            return float('inf')
        
        self.q_value = self.total_value / self.visit_count
        self.u_value = (exploration_param * self.prior_prob * 
                       np.sqrt(max(1, self.parent.visit_count)) / (1 + self.visit_count))
        return self.q_value + self.u_value
    
    def select_child(self):
        return max(self.children.values(), key=lambda child: child.ucb_score())
    
    def expand(self, valid_moves, action_probs):
        for move in valid_moves:
            if move not in self.children:
                # 获取该移动的先验概率
                from_pos, to_pos = move
                # 使用起始位置作为索引获取先验概率
                idx = from_pos[0] * 9 + from_pos[1]  # 索引到棋盘位置
                prob = action_probs[idx] if idx < len(action_probs) else 0.0
                self.children[move] = ChessMCTSNode(parent=self, prior_prob=prob, move=move)
    
    def update(self, value):
        self.visit_count += 1
        self.total_value += value
        if self.parent is not None:
            self.parent.update(-value)


class ChessMCTS:
    """中国象棋MCTS搜索"""

    def __init__(self, neural_net, num_simulations=75):  # 进一步减少模拟次数以提高速度
        self.neural_net = neural_net
        self.num_simulations = num_simulations
    
    def search(self, game):
        root = ChessMCTSNode()

        # 获取神经网络预测
        state_tensor = torch.FloatTensor(game.get_state()).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            policy, value = self.neural_net(state_tensor)
            policy = policy.cpu().numpy()[0]
            value = value.cpu().numpy()[0]

        # 获取有效移动
        valid_moves = game.get_all_valid_moves()
        if not valid_moves:
            return {}

        # 检查是否有直接获胜的移动（吃掉对方将/帅）
        winning_moves = []
        for move in valid_moves:
            from_pos, to_pos = move
            temp_game = self._copy_game(game)
            success = temp_game.make_move(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
            if success and temp_game.game_over and temp_game.winner == game.current_player:
                winning_moves.append(move)

        # 如果有直接获胜的移动，优先处理这些移动
        if winning_moves:
            # 为获胜移动分配极高访问次数，确保它们被优先选择
            result = {}
            for move in winning_moves:
                result[move] = self.num_simulations  # 设置为最大模拟次数
            # 对于非获胜移动，设置为0
            for move in valid_moves:
                if move not in winning_moves:
                    result[move] = 0
            return result

        # 为有效移动分配先验概率
        action_probs = {}
        for move in valid_moves:
            from_pos, to_pos = move
            idx = from_pos[0] * 9 + from_pos[1]
            action_probs[move] = policy[idx] if idx < len(policy) else 0.0

        # 归一化概率
        total_prob = sum(action_probs.values())
        if total_prob > 0:
            for move in action_probs:
                action_probs[move] /= total_prob

        # 扩展根节点
        root.expand(valid_moves, list(policy))  # 传递完整的策略数组

        # 运行模拟
        for _ in range(self.num_simulations):
            sim_game = self._copy_game(game)
            node = root
            search_path = [node]

            # 选择阶段
            while node.children and len(node.children) > 0:
                if node.visit_count == 0:
                    break
                node = node.select_child()
                success = sim_game.make_move(node.move[0][0], node.move[0][1],
                                            node.move[1][0], node.move[1][1])
                if not success or sim_game.game_over:
                    break
                search_path.append(node)

            # 扩展和评估阶段
            if not sim_game.game_over:
                # 获取当前状态的神经网络预测
                state_tensor = torch.FloatTensor(sim_game.get_state()).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    policy, value = self.neural_net(state_tensor)
                    policy = policy.cpu().numpy()[0]
                    value = value.cpu().numpy()[0]

                # 检查游戏是否结束
                if sim_game.game_over:
                    winner = sim_game.winner
                    if winner == sim_game.current_player:
                        value = 1.0
                    elif winner == -sim_game.current_player:
                        value = -1.0
                    else:
                        value = 0.0
                else:
                    # 扩展节点
                    valid_moves = sim_game.get_all_valid_moves()
                    if valid_moves:  # 确保有有效移动才扩展
                        node.expand(valid_moves, list(policy))
                    else:
                        # 如果没有有效移动，说明是僵局
                        value = 0.0  # 平局价值
            else:
                # 游戏结束，使用结果
                winner = sim_game.get_winner()
                if winner is None:
                    value = 0.0  # 如果无法确定获胜者，视为平局
                elif winner == sim_game.current_player:
                    value = 1.0
                elif winner == -sim_game.current_player:
                    value = -1.0
                else:
                    value = 0.0

            # 反向传播
            for path_node in search_path:
                path_node.update(value)

        # 返回访问次数
        return {move: child.visit_count for move, child in root.children.items()}
    
    def get_action_prob(self, game, temp=1.0):
        # 检查是否有直接获胜的移动
        valid_moves = game.get_all_valid_moves()
        winning_moves = []
        for move in valid_moves:
            from_pos, to_pos = move
            temp_game = self._copy_game(game)
            success = temp_game.make_move(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
            if success and temp_game.game_over and temp_game.winner == game.current_player:
                winning_moves.append(move)

        # 如果有直接获胜的移动，直接返回这些移动的概率为1
        if winning_moves:
            moves = winning_moves
            probs = [1.0 / len(winning_moves)] * len(winning_moves)  # 均匀分配概率给所有获胜移动
            return moves, probs

        # 否则进行正常MCTS搜索
        action_visits = self.search(game)

        if not action_visits:
            return None, None

        moves, visits = zip(*action_visits.items())
        visits = np.array(visits, dtype=np.float64)  # 使用双精度浮点数

        if temp == 0:
            best_move_idx = np.argmax(visits)
            best_move = moves[best_move_idx]
            probs = [0.0] * len(moves)
            probs[best_move_idx] = 1.0
        else:
            # 防止数值问题
            visits = np.maximum(visits, 0)  # 确保非负
            if np.sum(visits) == 0:  # 如果所有访问次数都是0
                # 返回均匀分布
                probs = np.ones(len(visits)) / len(visits)
            else:
                visits_power = np.power(visits, 1.0 / temp)
                # 检查是否有无穷大或NaN
                if np.any(np.isnan(visits_power)) or np.any(np.isinf(visits_power)):
                    # 如果有问题，使用原始访问次数作为概率
                    visits = visits + 1e-8  # 添加小的常数避免除零
                    probs = visits / np.sum(visits)
                else:
                    sum_visits_power = np.sum(visits_power)
                    if sum_visits_power == 0:
                        probs = np.ones(len(visits_power)) / len(visits_power)
                    else:
                        probs = visits_power / sum_visits_power

        # 确保概率和为1且无NaN
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            # 如果所有概率都为0，使用均匀分布
            probs = np.ones(len(probs)) / len(probs)

        return list(moves), list(probs)
    
    def _copy_game(self, game):
        """复制游戏状态"""
        new_game = ChineseChess()
        new_game.board = game.board.copy()
        new_game.current_player = game.current_player
        new_game.game_over = game.game_over
        new_game.winner = game.winner
        new_game.move_history = game.move_history[:]
        return new_game


class ChessTrainer:
    """中国象棋训练器"""
    
    def __init__(self, lr=0.001, batch_size=32):
        self.net = ChessNetwork()
        self.target_net = ChessNetwork()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = batch_size
        self.mse_loss = nn.MSELoss()
        
        # 更新目标网络
        self.update_target_net()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, pis, values = zip(*batch)

        # 转换数据格式
        states = torch.FloatTensor(np.array(states)).permute(0, 3, 1, 2)  # (batch, channels, height, width)
        target_pis = torch.FloatTensor(np.array(pis))
        target_values = torch.FloatTensor(np.array(values))

        pred_pis, pred_values = self.net(states)

        # 确保维度匹配
        policy_loss = -torch.mean(torch.sum(target_pis * torch.log(pred_pis + 1e-8), dim=1))
        value_loss = self.mse_loss(pred_values, target_values)
        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        # 梯度裁剪以避免梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), total_loss.item()
    
    def self_play(self, num_games=5, temp=1.0):
        """自我对弈生成训练数据"""
        training_data = []

        for game_num in range(num_games):
            print(f"正在进行第 {game_num + 1}/{num_games} 局自我对弈...")

            game = ChineseChess()
            mcts = ChessMCTS(self.net, num_simulations=75)  # 减少模拟次数以加快速度
            game_states = []
            pis = []
            move_history_detailed = []  # 记录详细的移动信息

            step = 0
            while not game.game_over:
                # 显示详细信息
                red_pieces = np.count_nonzero(game.board > 0)
                black_pieces = np.count_nonzero(game.board < 0)
                print(f"第 {step + 1} 步 | 红方剩余棋子: {red_pieces} | 黑方剩余棋子: {black_pieces} | 当前玩家: {'红方' if game.current_player == 1 else '黑方'}")
                
                # 显示当前棋盘
                if step % 20 == 0:  # 每20步显示一次棋盘
                    print("\n当前棋盘状态:")
                    game.print_board()

                # 获取MCTS建议的移动
                moves, probs = mcts.get_action_prob(game, temp)

                if not moves or len(moves) == 0:
                    print("没有有效移动，游戏结束")
                    break

                # 根据概率选择移动，增加更多随机性以避免平局
                # 但优先选择获胜移动
                if len(moves) == 1 and len(probs) == 1:
                    # 这意味着这是唯一的获胜移动，直接选择
                    move_idx = 0
                else:
                    # 使用Dirichlet噪声增加探索性
                    dir_epsilon = 0.3  # 增加探索参数，从0.25改为0.3
                    dir_alpha = 0.2  # 调整Dirichlet分布参数，从0.3改为0.2
                    
                    if np.random.rand() < dir_epsilon:  # 有一定概率随机选择
                        # 添加Dirichlet噪声以增加探索
                        noise = np.random.dirichlet([dir_alpha] * len(probs))
                        # 将probs转换为numpy数组进行数学运算
                        probs_array = np.array(probs)
                        mixed_probs = (1 - dir_epsilon) * probs_array + dir_epsilon * noise
                        move_idx = np.random.choice(len(moves), p=mixed_probs)
                    else:
                        move_idx = np.random.choice(len(moves), p=probs)
                move = moves[move_idx]

                # 记录状态和策略
                state = game.get_state()
                # 动作空间应该是所有可能的移动组合，这里简化为位置选择
                # 对于中国象棋，每个位置可以移动到其他位置，所以动作空间较大
                pi = np.zeros(10 * 9)  # 位置概率分布
                from_pos, to_pos = move
                from_idx = from_pos[0] * 9 + from_pos[1]
                to_idx = to_pos[0] * 9 + to_pos[1]

                # 简化：将移动表示为从某个位置出发的意图
                if from_idx < len(pi):
                    pi[from_idx] = probs[move_idx]  # 或者可以使用更复杂的编码方式

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

            # 游戏结束后，为每个状态分配价值标签（改进版）
            winner = game.winner if game.game_over else 0

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

                # 添加到经验回放缓冲区
                for state, pi, value in zip(game_states, pis, values):
                    self.replay_buffer.append((state, pi, value))

        return training_data
    
    def train(self, epochs=10, self_play_games_per_epoch=3, train_steps_per_epoch=50):
        """训练模型"""
        for epoch in range(epochs):
            print(f"\n=== 第 {epoch + 1}/{epochs} 轮训练 ===")
            
            # 自我对弈生成数据
            print("正在进行自我对弈...")
            self.self_play(num_games=self_play_games_per_epoch, temp=1.0)
            
            # 训练网络
            print("正在训练网络...")
            total_policy_loss = 0
            total_value_loss = 0
            total_loss = 0
            
            for step in range(train_steps_per_epoch):
                policy_loss, value_loss, loss = self.train_step()
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_loss += loss
                
                if step % 20 == 0:
                    print(f"步骤 {step}: 策略损失={policy_loss:.4f}, 价值损失={value_loss:.4f}, 总损失={loss:.4f}")
            
            avg_policy_loss = total_policy_loss / train_steps_per_epoch
            avg_value_loss = total_value_loss / train_steps_per_epoch
            avg_total_loss = total_loss / train_steps_per_epoch
            
            print(f"平均损失: 策略={avg_policy_loss:.4f}, 价值={avg_value_loss:.4f}, 总计={avg_total_loss:.4f}")
            
            # 更新目标网络
            self.update_target_net()
            
            print(f"第 {epoch + 1} 轮训练完成\n")
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'target_model_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"模型已保存至 {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
                self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
                self.target_net.load_state_dict(checkpoint['target_model_state_dict'], strict=False)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"模型已从 {filepath} 加载，可以继续训练")
            except Exception as e:
                print(f"加载模型时出错: {e}")
                print("将从随机初始化开始训练")
        else:
            print(f"模型文件不存在: {filepath}")
            # 如果文件不存在，仍可继续训练（从随机初始化开始）
            print("将从随机初始化开始训练")


def main():
    """主函数"""
    print("欢迎使用中国象棋AI训练系统!")
    
    trainer = ChessTrainer()
    
    # 快速训练测试
    print("开始快速训练测试...")
    trainer.train(epochs=2, self_play_games_per_epoch=2, train_steps_per_epoch=10)
    
    # 保存模型
    trainer.save_model("chinese_chess_model.pth")
    
    print("训练完成! 模型已保存。")


if __name__ == "__main__":
    main()