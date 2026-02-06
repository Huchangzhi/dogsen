"""
中国象棋AI主程序
整合所有功能的用户界面
"""
import os
import torch
import numpy as np
import pygame
from chinese_chess import ChineseChess
from chinese_chess_ai import ChessTrainer, ChessMCTS
from chinese_chess_gui import ChineseChessRenderer


def main_menu():
    """主菜单"""
    print("\n" + "="*50)
    print("欢迎使用中国象棋AI系统!")
    print("="*50)
    print("请选择功能:")
    print("1. 开始训练AI")
    print("2. 人机对战 (图形界面)")
    print("3. 观看AI自我对战 (图形界面)")
    print("4. 加载并测试模型")
    print("5. 退出")
    print("="*50)


def start_training():
    """开始训练"""
    print("\n开始训练中国象棋AI...")

    # 在开始训练前询问保存路径
    print("\n请为红方和黑方分别指定模型保存路径:")
    red_save_path = input("请输入红方模型保存路径 (默认: red_model.pth): ") or "red_model.pth"
    black_save_path = input("请输入黑方模型保存路径 (默认: black_model.pth): ") or "black_model.pth"

    # 询问是否为红方和黑方加载不同模型
    print("\n为红方选择模型:")
    red_model_path = input("请输入红方模型路径 (留空则使用随机模型): ")
    
    print("\n为黑方选择模型:")
    black_model_path = input("请输入黑方模型路径 (留空则使用随机模型): ")

    # 创建两个训练器，分别对应红方和黑方
    red_trainer = ChessTrainer()
    black_trainer = ChessTrainer()

    # 加载红方模型
    if red_model_path.strip():
        red_trainer.load_model(red_model_path)
        print(f"红方已加载模型: {red_model_path}")
    else:
        print("红方使用随机模型")

    # 加载黑方模型
    if black_model_path.strip():
        black_trainer.load_model(black_model_path)
        print(f"黑方已加载模型: {black_model_path}")
    else:
        print("黑方使用随机模型")

    epochs = int(input("请输入训练轮数 (默认5): ") or "5")
    games_per_epoch = int(input("请输入每轮自我对弈局数 (默认3): ") or "3")
    steps_per_epoch = int(input("请输入每轮训练步数 (默认20): ") or "20")

    # 修改训练逻辑，使用两个AI进行对战
    for epoch in range(epochs):
        print(f"\n=== 第 {epoch + 1}/{epochs} 轮训练 ===")

        # 自我对弈生成数据 - 使用两个不同的AI
        print("正在进行自我对弈 (红方 vs 黑方)...")
        
        for game_num in range(games_per_epoch):
            print(f"正在进行第 {game_num + 1}/{games_per_epoch} 局自我对弈...")
            
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
                # 显示详细信息
                red_pieces = np.count_nonzero(game.board > 0)
                black_pieces = np.count_nonzero(game.board < 0)
                print(f"第 {step + 1} 步 | 红方剩余棋子: {red_pieces} | 黑方剩余棋子: {black_pieces} | 当前玩家: {'红方' if game.current_player == 1 else '黑方'}")
                
                # 显示当前棋盘
                if step % 20 == 0:  # 每20步显示一次棋盘
                    print("\n当前棋盘状态:")
                    game.print_board()

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

        # 训练网络 - 分别训练两个AI
        print("正在训练红方AI...")
        total_policy_loss_red = 0
        total_value_loss_red = 0
        total_loss_red = 0

        for step in range(steps_per_epoch):
            policy_loss, value_loss, loss = red_trainer.train_step()
            total_policy_loss_red += policy_loss
            total_value_loss_red += value_loss
            total_loss_red += loss

        avg_policy_loss_red = total_policy_loss_red / steps_per_epoch if steps_per_epoch > 0 else 0
        avg_value_loss_red = total_value_loss_red / steps_per_epoch if steps_per_epoch > 0 else 0
        avg_total_loss_red = total_loss_red / steps_per_epoch if steps_per_epoch > 0 else 0

        print(f"红方平均损失: 策略={avg_policy_loss_red:.4f}, 价值={avg_value_loss_red:.4f}, 总计={avg_total_loss_red:.4f}")

        print("正在训练黑方AI...")
        total_policy_loss_black = 0
        total_value_loss_black = 0
        total_loss_black = 0

        for step in range(steps_per_epoch):
            policy_loss, value_loss, loss = black_trainer.train_step()
            total_policy_loss_black += policy_loss
            total_value_loss_black += value_loss
            total_loss_black += loss

        avg_policy_loss_black = total_policy_loss_black / steps_per_epoch if steps_per_epoch > 0 else 0
        avg_value_loss_black = total_value_loss_black / steps_per_epoch if steps_per_epoch > 0 else 0
        avg_total_loss_black = total_loss_black / steps_per_epoch if steps_per_epoch > 0 else 0

        print(f"黑方平均损失: 策略={avg_policy_loss_black:.4f}, 价值={avg_value_loss_black:.4f}, 总计={avg_total_loss_black:.4f}")

        # 更新目标网络
        red_trainer.update_target_net()
        black_trainer.update_target_net()

        print(f"第 {epoch + 1} 轮训练完成\n")

    # 保存最终模型 - 保存两个模型
    red_trainer.save_model(red_save_path)
    black_trainer.save_model(black_save_path)
    print(f"红方模型已保存至 {red_save_path}")
    print(f"黑方模型已保存至 {black_save_path}")


def play_against_ai():
    """人机对战"""
    print("\n开始人机对战...")

    model_path = input("请输入AI模型路径 (留空使用随机模型): ")

    # 初始化游戏
    game = ChineseChess()

    # 如果指定了模型路径，尝试加载AI
    if model_path and os.path.exists(model_path):
        trainer = ChessTrainer()
        trainer.load_model(model_path)
        ai_mcts = ChessMCTS(trainer.net, num_simulations=75)
        print(f"AI已加载模型: {model_path}")
    else:
        print("使用随机模型进行游戏")
        trainer = ChessTrainer()
        ai_mcts = ChessMCTS(trainer.net, num_simulations=50)

    # 启动图形界面
    renderer = ChineseChessRenderer()
    print("请稍候，启动图形界面...")

    # 修改renderer.run_game来支持人机对战
    selected_pos = None
    running = True
    clock = pygame.time.Clock()

    while running and not game.game_over:
        renderer.draw_board(game)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game.game_over and game.current_player == 1:  # 只有人类玩家回合才能操作
                row, col = renderer.get_square_from_mouse(event.pos)
                if row is not None and col is not None:
                    if selected_pos is None:
                        # 选择棋子
                        piece = game.board[row, col]
                        if piece != 0 and ((piece > 0 and game.current_player == 1) or (piece < 0 and game.current_player == -1)):
                            selected_pos = (row, col)
                    else:
                        # 移动棋子
                        from_row, from_col = selected_pos
                        success = game.make_move(from_row, from_col, row, col)
                        if success:
                            selected_pos = None  # 成功移动后取消选择
                        else:
                            # 如果移动无效，检查是否点击了另一个己方棋子
                            piece = game.board[row, col]
                            if piece != 0 and ((piece > 0 and game.current_player == 1) or (piece < 0 and game.current_player == -1)):
                                selected_pos = (row, col)
                            else:
                                selected_pos = None  # 取消选择

        # 如果当前是AI回合，AI自动下棋
        if not game.game_over and game.current_player == -1:  # AI（黑方）
            print("AI思考中...")
            moves, probs = ai_mcts.get_action_prob(game, temp=0.1)
            if moves and len(moves) > 0:
                # 如果是获胜移动，直接选择（当只有一个获胜移动时）
                if len(moves) == 1 and len(probs) == 1:
                    # 这意味着这是唯一的获胜移动，直接选择
                    move_idx = 0
                else:
                    # 使用更多随机性，避免过于保守
                    dir_epsilon = 0.2  # 探索参数
                    dir_alpha = 0.2  # Dirichlet分布参数
                    
                    if np.random.rand() < dir_epsilon:  # 有一定概率随机选择
                        # 添加Dirichlet噪声以增加探索
                        noise = np.random.dirichlet([dir_alpha] * len(probs))
                        # 将probs转换为numpy数组进行数学运算
                        probs_array = np.array(probs)
                        mixed_probs = (1 - dir_epsilon) * probs_array + dir_epsilon * noise
                        move_idx = np.random.choice(len(moves), p=mixed_probs)
                    else:
                        move_idx = np.argmax(probs)  # 选择最高概率的移动
                move = moves[move_idx]
                from_pos, to_pos = move
                game.make_move(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
                print(f"AI移动: {from_pos} -> {to_pos}")
                # 确保AI移动后切换到人类玩家回合
                selected_pos = None

        # 如果有选中的棋子，高亮显示
        if selected_pos:
            row, col = selected_pos
            x = renderer.margin_x + col * renderer.square_size
            y = renderer.margin_y + row * renderer.square_size
            pygame.draw.rect(renderer.screen, (255, 255, 0), (x, y, renderer.square_size, renderer.square_size), 3)
            pygame.display.flip()

        clock.tick(60)

    # 显示最终结果
    renderer.draw_board(game)
    print("游戏结束!")
    if game.winner == 1:
        print("人类玩家获胜!")
    elif game.winner == -1:
        print("AI获胜!")
    else:
        print("平局!")

    input("按Enter键返回主菜单...")
    pygame.quit()


def watch_ai_self_play():
    """观看AI自我对战"""
    print("\n开始AI自我对战...")

    # 询问是否为红方和黑方加载不同模型
    print("\n为红方选择模型:")
    red_model_path = input("请输入红方模型路径 (留空则使用随机模型): ")
    
    print("\n为黑方选择模型:")
    black_model_path = input("请输入黑方模型路径 (留空则使用随机模型): ")

    # 初始化游戏
    game = ChineseChess()

    # 加载红方AI
    if red_model_path and os.path.exists(red_model_path):
        red_trainer = ChessTrainer()
        red_trainer.load_model(red_model_path)
        red_mcts = ChessMCTS(red_trainer.net, num_simulations=75)
        print(f"红方AI已加载模型: {red_model_path}")
    else:
        print("红方AI使用随机模型")
        red_trainer = ChessTrainer()
        red_mcts = ChessMCTS(red_trainer.net, num_simulations=50)

    # 加载黑方AI
    if black_model_path and os.path.exists(black_model_path):
        black_trainer = ChessTrainer()
        black_trainer.load_model(black_model_path)
        black_mcts = ChessMCTS(black_trainer.net, num_simulations=75)
        print(f"黑方AI已加载模型: {black_model_path}")
    else:
        print("黑方AI使用随机模型")
        black_trainer = ChessTrainer()
        black_mcts = ChessMCTS(black_trainer.net, num_simulations=50)

    # 启动图形界面
    renderer = ChineseChessRenderer()
    print("请稍候，启动图形界面...")

    running = True
    clock = pygame.time.Clock()

    while running and not game.game_over:
        renderer.draw_board(game)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # AI自动下棋
        if not game.game_over:
            if game.current_player == 1:  # 红方AI
                print("红方AI思考中...")
                # 使用更高的温度参数增加随机性，避免平局
                moves, probs = red_mcts.get_action_prob(game, temp=1.0)
                if moves and len(moves) > 0:
                    # 如果是获胜移动，直接选择（当只有一个获胜移动时）
                    if len(moves) == 1 and len(probs) == 1:
                        # 这意味着这是唯一的获胜移动，直接选择
                        move_idx = 0
                    else:
                        # 引入更多随机性选择移动，使用Dirichlet噪声
                        dir_epsilon = 0.35  # 增加探索参数，从0.3改为0.35
                        dir_alpha = 0.2  # 调整Dirichlet分布参数，从0.3改为0.2

                        if np.random.rand() < dir_epsilon:  # 有一定概率随机选择
                            # 添加Dirichlet噪声以增加探索
                            noise = np.random.dirichlet([dir_alpha] * len(probs))
                            # 将probs转换为numpy数组进行数学运算
                            probs_array = np.array(probs)
                            mixed_probs = (1 - dir_epsilon) * probs_array + dir_epsilon * noise
                            move_idx = np.random.choice(len(moves), p=mixed_probs)
                        elif np.random.rand() < 0.6:  # 60% 概率选择最高概率的移动
                            move_idx = np.argmax(probs)
                        else:  # 40% 概率根据概率分布随机选择
                            move_idx = np.random.choice(len(moves), p=probs)

                    move = moves[move_idx]
                    from_pos, to_pos = move
                    game.make_move(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
                    print(f"红方AI移动: {from_pos} -> {to_pos}")
            else:  # 黑方AI
                print("黑方AI思考中...")
                # 使用更高的温度参数增加随机性，避免平局
                moves, probs = black_mcts.get_action_prob(game, temp=1.0)
                if moves and len(moves) > 0:
                    # 如果是获胜移动，直接选择（当只有一个获胜移动时）
                    if len(moves) == 1 and len(probs) == 1:
                        # 这意味着这是唯一的获胜移动，直接选择
                        move_idx = 0
                    else:
                        # 引入更多随机性选择移动，使用Dirichlet噪声
                        dir_epsilon = 0.35  # 增加探索参数，从0.3改为0.35
                        dir_alpha = 0.2  # 调整Dirichlet分布参数，从0.3改为0.2

                        if np.random.rand() < dir_epsilon:  # 有一定概率随机选择
                            # 添加Dirichlet噪声以增加探索
                            noise = np.random.dirichlet([dir_alpha] * len(probs))
                            # 将probs转换为numpy数组进行数学运算
                            probs_array = np.array(probs)
                            mixed_probs = (1 - dir_epsilon) * probs_array + dir_epsilon * noise
                            move_idx = np.random.choice(len(moves), p=mixed_probs)
                        elif np.random.rand() < 0.6:  # 60% 概率选择最高概率的移动
                            move_idx = np.argmax(probs)
                        else:  # 40% 概率根据概率分布随机选择
                            move_idx = np.random.choice(len(moves), p=probs)

                    move = moves[move_idx]
                    from_pos, to_pos = move
                    game.make_move(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
                    print(f"黑方AI移动: {from_pos} -> {to_pos}")

        clock.tick(1)  # 每秒1步，便于观察

    # 显示最终结果
    renderer.draw_board(game)
    print("游戏结束!")
    if game.winner == 1:
        print("红方获胜!")
    elif game.winner == -1:
        print("黑方获胜!")
    else:
        print("平局!")

    input("按Enter键返回主菜单...")


def load_and_test_model():
    """加载并测试模型"""
    print("\n加载并测试模型...")
    
    model_path = input("请输入模型路径: ")
    if not os.path.exists(model_path):
        print("模型文件不存在!")
        return
    
    trainer = ChessTrainer()
    trainer.load_model(model_path)
    
    print(f"模型加载成功!")
    print(f"参数数量: {sum(p.numel() for p in trainer.net.parameters()):,}")
    
    # 简单测试模型
    game = ChineseChess()
    dummy_input = torch.FloatTensor(game.get_state()).permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        policy, value = trainer.net(dummy_input)
    
    print(f"前向传播测试成功!")
    print(f"策略输出形状: {policy.shape}")
    print(f"价值输出: {value.item():.4f}")


def main():
    """主函数"""
    while True:
        main_menu()
        choice = input("\n请输入选择 (1-5): ")
        
        if choice == "1":
            start_training()
        elif choice == "2":
            play_against_ai()
        elif choice == "3":
            watch_ai_self_play()
        elif choice == "4":
            load_and_test_model()
        elif choice == "5":
            print("感谢使用中国象棋AI系统，再见!")
            break
        else:
            print("无效选择，请重新输入!")


if __name__ == "__main__":
    main()