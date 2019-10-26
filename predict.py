from predict_functions import load_checkpoint, predict, sanity_check
import argparse 

def get_args():
    parser = argparse.ArgumentParser(description= "Predict image class")
    parser.add_argument("image_path", help= "Path to image", type=str)
    parser.add_argument("checkpoint", help= "path to checkpoint", default="checkpoint.pth", type=str)
    parser.add_argument("--top_k", help= "top k classes of image", default=5, type=int)
    parser.add_argument("--category_names", help="category names of top k classes", action="store_true")
    parser.add_argument("--gpu", help="use gpu for inference", action="store_true")
    
    args = parser.parse_args()
    
    return args




def main():
    args = get_args()
    model = load_checkpoint(args.checkpoint)
    prob, label = predict(args.image_path, model, args.top_k, args.gpu)
    print("predeicted categories are: ", label, " with probabilities ", prob, ", respectivly")
    if args.category_names:
        cat_names = sanity_check(prob, label)
        print("category names: ", cat_names)   

     

if __name__ == "__main__" :
    main()