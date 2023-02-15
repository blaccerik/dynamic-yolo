from flask import Blueprint

REQUEST_API = Blueprint('upload',__name__)



# @REQUEST_API.route('/', methods=['POST'])
# def create_record():
#     """
#
#     """
#     data = User().load(request.json)
#     name = data["name"]
#     msg = create_user(name)
#     if msg == "error":
#         return jsonify({'error': 'User with that name already exists'}), 409
#
#     return jsonify({'message': 'User created successfully'}), 201

